"""Replace every feed-forward block of a ModelOpt FP8 lean-encoder ONNX with the FFNFp8 TensorRT plugin node.

Pattern per block (ModelOpt export, no biases):
  Q_x -> DQ_x -> MatMul1 <- Transpose <- DQ_w1 <- Q_w1(W1);  MatMul1 -> Sigmoid, Mul -> Q_h -> DQ_h -> MatMul2
  MatMul2 <- Transpose <- DQ_w2 <- Q_w2(W2)
becomes FFNFp8(Q_x output) -> MatMul2's output name, with W1, W2 quantised offline exactly as TensorRT would
(w / s, saturated to +-448, round to nearest e4m3) and alpha1 = s_x * s_w1, oscale = 1 / s_h, alpha2 = s_h * s_w2.
Ablation: build from the input ONNX instead (no plugin).
With FFN_RESIDUAL=1 the following half-step residual add (Mul by a constant, then Add with the residual) is folded
into the plugin too (second input = residual, attribute res_scale), saving the FF output's write and re-read.
usage: ffn_surgery.py IN.onnx OUT.onnx"""
from __future__ import annotations

import os
import re
import sys

import numpy as np
import onnx
import torch
from onnx import helper, numpy_helper

src, dst = sys.argv[1], sys.argv[2]
os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
m = onnx.load(src)
g = m.graph
inits = {i.name: i for i in g.initializer}
prod = {o: n for n in g.node for o in n.output}
cons: dict[str, list] = {}
for n in g.node:
    for i in n.input:
        cons.setdefault(i, []).append(n)


def up(name: str, casts: list):
    """Producer of a tensor, stepping through Cast nodes (fp16 exports wrap Q/DQ in casts); casts are collected."""
    n = prod[name]
    while n.op_type == "Cast":
        casts.append(n)
        n = prod[n.input[0]]
    return n


def down(name: str, casts: list):
    """Single consumer of a tensor, stepping through Cast nodes."""
    n = cons[name][0]
    while n.op_type == "Cast":
        casts.append(n)
        n = cons[n.output[0]][0]
    return n


def const_val(name: str) -> float:
    n = prod[name]
    assert n.op_type == "Constant", n
    return float(numpy_helper.to_array(n.attribute[0].t).reshape(-1)[0])


def weight_path(matmul):
    """MatMul's B input: Transpose <- DQ <- Q(W, s). Returns (nodes, W name, s)."""
    casts: list = []
    tr = up(matmul.input[1], casts)
    dq = up(tr.input[0], casts)
    q = up(dq.input[0], casts)
    assert (tr.op_type, dq.op_type, q.op_type) == ("Transpose", "TRT_FP8DequantizeLinear", "TRT_FP8QuantizeLinear")
    w = q.input[0]
    while w not in inits:                                          # weight may itself pass through a Cast
        c = prod[w]
        assert c.op_type == "Cast", c
        casts.append(c)
        w = c.input[0]
    return [tr, dq, q, prod[dq.input[1]], prod[q.input[1]], *casts], w, const_val(q.input[1])


def quant(w: np.ndarray, s: float) -> np.ndarray:
    t = torch.from_numpy(w.astype(np.float32)) / s
    return t.clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.int8).numpy()


RESIDUAL = os.environ.get("FFN_RESIDUAL", "0") == "1"
SPARSE = os.environ.get("FFN_SPARSE", "0") == "1"
FP4 = os.environ.get("FFN_FP4", "0") == "1"                   # NVFP4 FF (plugin fp4 mode): fp16 input, no FP8 quantise
FP4_SKIP = {int(v) for v in os.environ.get("FFN_FP4_SKIP", "").split(",") if v}   # layers kept FP8 (e.g. 0,1,22,23)
n_fp4 = 0           # weights 2:4-pruned (finetune_stride --sparse24)


def is_24(q: np.ndarray) -> bool:
    """Every group of 4 along K has at least 2 exact zeros (FP8 bytes 0x00 or 0x80)."""
    z = (q.view(np.uint8) & 0x7F) == 0
    return bool((z.reshape(q.shape[0], -1, 4).sum(-1) >= 2).all())


def scalar_of(name: str):
    """Value of a scalar constant input (Constant node or initializer), else None."""
    if name in inits:
        a = numpy_helper.to_array(inits[name])
    elif name in prod and prod[name].op_type == "Constant":
        a = numpy_helper.to_array(prod[name].attribute[0].t)
    else:
        return None
    return float(a.reshape(-1)[0]) if a.size == 1 else None


remove, new_nodes, drop_inits = set(), [], set()
pat = re.compile(r"/feed_forward[12]/linear1(_\d+)?/MatMul$")
count = 0
for mm1 in [n for n in g.node if pat.search(n.name)]:
    casts: list = []
    dq_x = up(mm1.input[0], casts)
    q_x = prod[dq_x.input[0]]
    assert dq_x.op_type == "TRT_FP8DequantizeLinear" and q_x.op_type == "TRT_FP8QuantizeLinear"
    assert len(cons[dq_x.output[0]]) == 1, "DQ_x feeds more than the FF"
    s_x = const_val(q_x.input[1])
    w1_nodes, w1_name, s_w1 = weight_path(mm1)
    def bias_after(mm):
        """(Add node, bias vector) if the MatMul's only consumer adds a constant vector (biased FF linears)."""
        c = cons[mm.output[0]]
        if len(c) == 1 and c[0].op_type == "Add":
            other = c[0].input[1] if c[0].input[0] == mm.output[0] else c[0].input[0]
            if other in inits:
                return c[0], numpy_helper.to_array(inits[other]).astype(np.float32).reshape(-1)
            if other in prod and prod[other].op_type == "Constant":
                return c[0], numpy_helper.to_array(prod[other].attribute[0].t).astype(np.float32).reshape(-1)
        return None, None
    add1, b1 = bias_after(mm1)
    h_out = add1.output[0] if add1 is not None else mm1.output[0]
    sig, mul = sorted(cons[h_out], key=lambda n: n.op_type != "Sigmoid")
    assert (sig.op_type, mul.op_type) == ("Sigmoid", "Mul")
    q_h = down(mul.output[0], casts)
    dq_h = down(q_h.output[0], casts)
    mm2 = down(dq_h.output[0], casts)
    add2, b2 = bias_after(mm2)
    if (add1 is None) != (add2 is None):
        raise SystemExit("FF block with a bias on only one linear: not supported")
    assert (q_h.op_type, dq_h.op_type, mm2.op_type) == ("TRT_FP8QuantizeLinear", "TRT_FP8DequantizeLinear", "MatMul")
    s_h = const_val(q_h.input[1])
    w2_nodes, w2_name, s_w2 = weight_path(mm2)
    W1 = numpy_helper.to_array(inits[w1_name])                    # [N1, K1] (nn.Linear layout)
    W2 = numpy_helper.to_array(inits[w2_name])                    # [N2, N1]
    N1, K1 = W1.shape
    N2 = W2.shape[0]
    node = helper.make_node(
        "FFNFp8", [q_x.output[0]], [mm2.output[0]], name=mm1.name.replace("/linear1", "/ffn_fp8").replace("/MatMul", ""),
        w1=numpy_helper.from_array(quant(W1, s_w1)), w2=numpy_helper.from_array(quant(W2, s_w2)),
        dims=[int(K1), int(N1), int(N2)], alpha1=s_x * s_w1, oscale=1.0 / s_h, alpha2=s_h * s_w2,
        plugin_version="1", plugin_namespace="")
    lm = re.search(r"layers\.(\d+)\.", w1_name)
    if FP4 and FP4_SKIP and lm is None:
        raise SystemExit(f"FFN_FP4_SKIP: no layer index in {w1_name}")
    if FP4 and not (lm and int(lm.group(1)) in FP4_SKIP):
        n_fp4 += 1
        if add1 is not None or SPARSE:
            raise SystemExit("FFN_FP4: biases / sparse24 not supported")
        node.input[0] = q_x.input[0]                               # the plugin quantises to NVFP4 itself
        node.attribute.extend([helper.make_attribute("fp4", 1), helper.make_attribute("sx", s_x)])
        if len(cons[q_x.output[0]]) == 1:
            remove |= {id(q_x), id(prod[q_x.input[1]])}
    if SPARSE:
        q1, q2 = quant(W1, s_w1), quant(W2, s_w2)
        if not (is_24(q1) and is_24(q2)):
            raise SystemExit(f"FFN_SPARSE: {mm1.name} weights are not 2:4 sparse")
        node.attribute.append(helper.make_attribute("sparse24", 1))
    last = add2 if add2 is not None else mm2                      # FF block output node
    if add2 is not None:
        node.output[0] = add2.output[0]
        node.attribute.extend([helper.make_attribute("b1", b1.tolist()), helper.make_attribute("b2", b2.tolist())])
        remove |= {id(add1), id(add2)}
    anchor = last
    if RESIDUAL:
        rmul = cons[last.output[0]]
        assert len(rmul) == 1 and rmul[0].op_type == "Mul", [n.op_type for n in rmul]
        rmul = rmul[0]
        k = scalar_of(rmul.input[1] if rmul.input[0] == last.output[0] else rmul.input[0])
        radd = cons[rmul.output[0]]
        assert k is not None and len(radd) == 1 and radd[0].op_type == "Add"
        radd = radd[0]
        res = radd.input[1] if radd.input[0] == rmul.output[0] else radd.input[0]
        node.input.append(res)
        node.output[0] = radd.output[0]
        node.attribute.append(helper.make_attribute("res_scale", k))
        remove |= {id(rmul), id(radd)}
        anchor = radd
    new_nodes.append((anchor, node))
    for n in [dq_x, prod[dq_x.input[1]], mm1, sig, mul, q_h, prod[q_h.input[1]], dq_h, prod[dq_h.input[1]], mm2,
              *w1_nodes, *w2_nodes, *casts]:
        remove.add(id(n))
    drop_inits |= {w1_name, w2_name}
    count += 1

# rebuild the node list in order: the plugin node goes where MatMul2 was (keeps topological order)
repl = {id(mm2): node for mm2, node in new_nodes}
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
print(f"replaced {count} FF blocks ({n_fp4} NVFP4) -> {dst}", flush=True)
