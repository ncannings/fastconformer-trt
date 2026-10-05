"""For a 2:4-pruned model (finetune_stride --sparse24): replace the encoder's conv-module pointwise linears (pw1, pw2)
and attention output projections (linear_out) with the SpLinearFp8 plugin (CUTLASS sm120 sparse), folding the
residual Add that follows pw2 and linear_out. Each weight is quantised as TensorRT would (w / s, saturated, e4m3) and
checked to be 2:4 along K before replacement (refuses otherwise). Run after ffn_surgery.py and qkv_surgery.py.
SP_DENSE=1: unpruned (stock-weight) model; only linear_out and pw2 (the ones followed by a residual Add) are replaced,
with dense FP8 weights (plugin field dense=1), so the residual add moves into the GEMM epilogue; no 2:4 check.
usage: [SP_DENSE=1] sparse_surgery.py IN.onnx OUT.onnx"""
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


def const_arr(name):
    """Value of an initializer or Constant-node output, else None."""
    if name in inits:
        return numpy_helper.to_array(inits[name])
    n = prod.get(name)
    if n is not None and n.op_type == "Constant":
        return numpy_helper.to_array(n.attribute[0].t)
    return None


def is_24(q):
    z = (q.view(np.uint8) & 0x7F) == 0
    return bool((z.reshape(q.shape[0], -1, 4).sum(-1) >= 2).all())


DENSE = os.environ.get("SP_DENSE", "0") == "1"
pat = re.compile(r"^(pw1\.\d+|pw2\.\d+|layers\.\d+\.self_attn\.linear_out)\.weight$")
remove, repl, drop_inits, n_rep, n_res = set(), {}, set(), 0, 0
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
    if not pat.match(wname) or (DENSE and wname.startswith("pw1")):
        continue
    dqx = up(mm.input[0], casts)
    qx = prod[dqx.input[0]]
    assert dqx.op_type == "TRT_FP8DequantizeLinear" and qx.op_type == "TRT_FP8QuantizeLinear", (dqx.op_type, qx.op_type)
    s_x, s_w = const_val(qx.input[1]), const_val(qw.input[1])
    W = numpy_helper.to_array(inits[wname]).astype(np.float32)                    # [N, K]
    q = torch.from_numpy(W / s_w).clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.int8).numpy()
    if not DENSE and not is_24(q):
        raise SystemExit(f"{wname} is not 2:4 sparse: refusing")
    N, K = W.shape
    inputs, out_name, anchor, res_scale = [qx.output[0]], mm.output[0], mm, 0.0
    c = cons[mm.output[0]]
    bias, cur = None, mm.output[0]
    if DENSE and len(c) == 1 and c[0].op_type == "Add":                            # bias add (models with use_bias)
        o = c[0].input[1] if c[0].input[0] == cur else c[0].input[0]
        if const_arr(o) is not None:
            bias = const_arr(o).astype(np.float32).reshape(-1)
            assert bias.size == W.shape[0], (wname, bias.shape)
            remove.add(id(c[0]))
            cur = c[0].output[0]
            c = cons[cur]
    if not wname.startswith("pw1") and len(c) == 1 and c[0].op_type == "Add":       # residual add follows
        add = c[0]
        res = add.input[1] if add.input[0] == cur else add.input[0]
        if const_arr(res) is not None:
            raise SystemExit(f"{wname}: the Add after it has a constant input, not a residual")
        inputs.append(res)
        out_name, anchor, res_scale = add.output[0], add, 1.0
        remove.add(id(add))
        n_res += 1
    elif bias is not None:
        raise SystemExit(f"{wname}: bias without a residual Add is not supported")
    if DENSE and res_scale == 0.0:
        raise SystemExit(f"SP_DENSE: {wname} has no residual Add to fold")
    node = helper.make_node("SpLinearFp8", inputs, [out_name], name=mm.name.replace("/MatMul", "/sp24"),
                            w=numpy_helper.from_array(q), dims=[int(K), int(N)], alpha=s_x * s_w,
                            res_scale=res_scale, dense=int(DENSE), plugin_version="1", plugin_namespace="",
                            **({"bias": bias.tolist()} if bias is not None else {}))
    repl[id(anchor)] = node
    remove |= {id(mm), id(tr), id(dqw), id(qw), id(prod[dqw.input[1]]), id(prod[qw.input[1]]), *map(id, casts)}
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
print(f"{'dense' if DENSE else 'sparse'}: replaced {n_rep} linears ({n_res} with residual folded) -> {dst}", flush=True)
