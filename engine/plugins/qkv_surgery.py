"""Feed each QKVHeads plugin node the FP8 output of its input quantiser (the export passes the dequantised tensor),
removing the then-unused DequantizeLinear, and attach its weights, bias, alpha and dims from the side file
lean_export.py writes next to the FP8 ONNX (*.qkv.npz). No-op for graphs without QKVHeads.
usage: qkv_surgery.py IN.onnx OUT.onnx SIDE.npz"""
import os
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper

src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src)
g = m.graph
prod = {o: n for n in g.node for o in n.output}
cons = {}
for n in g.node:
    for i in n.input:
        cons.setdefault(i, []).append(n)
drop, k = set(), 0
side = np.load(sys.argv[3]) if len(sys.argv) > 3 and os.path.exists(sys.argv[3]) else None
for n in [n for n in g.node if n.op_type == "QKVHeads"]:
    n.domain = ""                                                  # exported in a custom domain (exporter schema lookup)
    li = [helper.get_attribute_value(a) for a in n.attribute if a.name == "layer"][0]
    keep_attrs = [a for a in n.attribute if a.name != "layer"]
    del n.attribute[:]
    n.attribute.extend(keep_attrs)
    if os.environ.get("QKV_SPARSE", "0") == "1":                # 2:4-pruned model: sparse kernels, checked here
        q = side[f"w{li}"]
        z = (q.view(np.uint8) & 0x7F) == 0
        if not (z.reshape(q.shape[0], -1, 4).sum(-1) >= 2).all():
            raise SystemExit(f"QKV_SPARSE: layer {li} weights are not 2:4")
        n.attribute.append(helper.make_attribute("sparse24", 1))
    n.attribute.extend([helper.make_attribute("w", numpy_helper.from_array(side[f"w{li}"])),
                        helper.make_attribute("bias", side[f"b{li}"].tolist()),
                        helper.make_attribute("alpha", [float(a) for a in side[f"a{li}"]]),
                        helper.make_attribute("dims", side[f"d{li}"].tolist())])
    dq = prod[n.input[0]]
    while dq.op_type == "Cast":                                    # fp16 exports may cast around the Q/DQ pair
        dq = prod[dq.input[0]]
    assert dq.op_type == "TRT_FP8DequantizeLinear", dq.op_type
    n.input[0] = dq.input[0]
    if all(c is n or id(c) in drop for c in cons[dq.output[0]]):
        drop.add(id(dq))
    k += 1
import json
here = os.path.dirname(os.path.abspath(__file__))
ra = 0
for n in [n for n in g.node if n.op_type == "RelPosAttn"]:                 # attach the compiled attention kernel
    n.domain = ""
    dk = ([helper.get_attribute_value(a) for a in n.attribute if a.name == "dk"] or [128])[0]
    stem = "relpos" if dk == 128 else f"relpos{dk}"                         # cubins are compiled per head dim
    meta = json.load(open(os.path.join(here, stem + ".json")))
    if meta.get("DK", 128) != dk:
        raise SystemExit(f"{stem}.cubin is compiled for head dim {meta.get('DK', 128)}, node needs {dk}")
    cubin = np.frombuffer(open(os.path.join(here, stem + ".cubin"), "rb").read(), dtype=np.int8)
    n.attribute.extend([helper.make_attribute("cubin", numpy_helper.from_array(cubin)),
                        helper.make_attribute("kname", meta["name"]),
                        helper.make_attribute("smem", int(meta["shared"])),
                        helper.make_attribute("warps", int(meta["num_warps"])),
                        helper.make_attribute("bm", int(meta["BM"]))])
    ra += 1
print(f"attached the attention cubin to {ra} RelPosAttn nodes", flush=True)
for n in g.node:                                                           # other trt:: plugin ops (SubConv02)
    if n.domain == "trt":
        n.domain = ""
keep = [n for n in g.node if id(n) not in drop]
del g.node[:]
g.node.extend(keep)
onnx.save(m, dst, save_as_external_data=True, location=os.path.basename(dst) + ".data")
print(f"rewired {k} QKVHeads nodes -> {dst}", flush=True)
