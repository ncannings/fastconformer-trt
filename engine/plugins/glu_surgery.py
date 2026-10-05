"""Replace each conv-module pw1 MatMul + GLU (Split -> Sigmoid(gate) -> Mul(value, .)) with the PwGluFp8 plugin
(glu_plugin.cpp), same FP8 weights (quantised as TensorRT would: w / s, saturated, e4m3) and the FP8 input of the
MatMul's quantiser. Refuses on any unexpected pattern (biases, other split axes). Run after ffn/qkv/sparse surgery.
usage: glu_surgery.py IN.onnx OUT.onnx"""
from __future__ import annotations

import os
import re
import sys

import numpy as np
import onnx
import torch
from onnx import helper, numpy_helper

src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src)
g = m.graph
inits = {i.name: i for i in g.initializer}
prod = {o: n for n in g.node for o in n.output}
cons: dict[str, list] = {}
for n in g.node:
    for i in n.input:
        cons.setdefault(i, []).append(n)


def up(name, casts):
    n = prod[name]
    while n.op_type == "Cast":
        casts.append(n)
        n = prod[n.input[0]]
    return n


def const_val(name):
    return float(numpy_helper.to_array(prod[name].attribute[0].t).reshape(-1)[0])


pat = re.compile(r"^pw1\.\d+\.weight$")
remove, repl, drop_inits, n_rep = set(), {}, set(), 0
for mm in [n for n in g.node if n.op_type == "MatMul"]:
    casts: list = []
    try:
        tr = up(mm.input[1], casts)
        dqw = up(tr.input[0], casts)
        qw = up(dqw.input[0], casts)
    except KeyError:
        continue
    if tr.op_type != "Transpose" or qw.op_type != "TRT_FP8QuantizeLinear" or qw.input[0] not in inits:
        continue
    wname = qw.input[0]
    if not pat.match(wname):
        continue
    c = cons[mm.output[0]]
    if len(c) != 1 or c[0].op_type != "Split":
        raise SystemExit(f"{wname}: expected a Split after the MatMul, got {[x.op_type for x in c]}")
    sp = c[0]
    ax = {a.name: helper.get_attribute_value(a) for a in sp.attribute}.get("axis")
    if ax not in (-1, 2) or len(sp.output) != 2:
        raise SystemExit(f"{wname}: Split axis {ax} / {len(sp.output)} outputs")
    sg = cons[sp.output[1]]
    if len(sg) != 1 or sg[0].op_type != "Sigmoid":
        raise SystemExit(f"{wname}: gate half does not go to one Sigmoid")
    mul = [x for x in cons[sg[0].output[0]] if x.op_type == "Mul"]
    if len(mul) != 1 or sp.output[0] not in mul[0].input or len(cons[sp.output[0]]) != 1:
        raise SystemExit(f"{wname}: GLU product not found")
    mul = mul[0]
    dqx = up(mm.input[0], casts)
    qx = prod[dqx.input[0]]
    assert dqx.op_type == "TRT_FP8DequantizeLinear" and qx.op_type == "TRT_FP8QuantizeLinear", (dqx.op_type, qx.op_type)
    s_x, s_w = const_val(qx.input[1]), const_val(qw.input[1])
    W = numpy_helper.to_array(inits[wname]).astype(np.float32)                    # [2N, K]
    q = torch.from_numpy(W / s_w).clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.int8).numpy()
    N2, K = W.shape
    node = helper.make_node("PwGluFp8", [qx.output[0]], [mul.output[0]], name=mm.name.replace("/MatMul", "/glu"),
                            w=numpy_helper.from_array(q), dims=[int(K), int(N2 // 2)], alpha=s_x * s_w,
                            plugin_version="1", plugin_namespace="")
    repl[id(mul)] = node
    remove |= {id(mm), id(tr), id(dqw), id(qw), id(prod[dqw.input[1]]), id(prod[qw.input[1]]), id(sp), id(sg[0]),
               *map(id, casts)}
    if len(cons[dqx.output[0]]) == 1:
        remove |= {id(dqx), id(prod[dqx.input[1]])}
    drop_inits.add(wname)
    n_rep += 1
nodes = []
for n in g.node:
    if id(n) in repl:
        nodes.append(repl[id(n)])
    elif id(n) not in remove:
        nodes.append(n)
del g.node[:]
g.node.extend(nodes)
keep = [i for i in g.initializer if i.name not in drop_inits]
del g.initializer[:]
g.initializer.extend(keep)
onnx.save(m, dst, save_as_external_data=True, location=os.path.basename(dst) + ".data")
print(f"glu: replaced {n_rep} pw1+GLU -> {dst}", flush=True)
