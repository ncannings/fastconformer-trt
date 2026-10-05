"""Replace conv.0 -> ReLU -> conv.2 (depthwise) of the encoder's subsampling with the SubConv02 TensorRT plugin
(sub_plugin.cu), same weights. Run after ffn_surgery.py or on any lean-encoder ONNX.
usage: sub_surgery.py IN.onnx OUT.onnx"""
from __future__ import annotations

import os
import sys

import onnx
from onnx import helper, numpy_helper

src, dst = sys.argv[1], sys.argv[2]
os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
m = onnx.load(src)
g = m.graph
inits = {i.name: i for i in g.initializer}
byname = {n.name: n for n in g.node}
c0, relu, c2 = byname["/conv.0/Conv"], byname["/conv.1/Relu"], byname["/conv.2/Conv"]
assert relu.input[0] == c0.output[0] and c2.input[0] == relu.output[0]
for c, grp in ((c0, 1), (c2, None)):
    at = {a.name: helper.get_attribute_value(a) for a in c.attribute}
    assert at["kernel_shape"] == [3, 3] and at["strides"] == [2, 2] and at["pads"] == [1, 1, 1, 1], at
w = {k: numpy_helper.to_array(inits[n]).astype("float32").reshape(-1)
     for k, n in (("w0", c0.input[1]), ("b0", c0.input[2]), ("w2", c2.input[1]), ("b2", c2.input[2]))}
NHWC = os.environ.get("SUB_NHWC", "1") == "1"      # plugin writes NHWC; a Transpose restores NCHW for the graph
PW3 = os.environ.get("SUB_PW3", "0") == "1"        # also fuse conv.3 (1x1 + bias) and its ReLU (tensor cores)
end, gone = c2, set()
if PW3:
    assert NHWC, "SUB_PW3 needs SUB_NHWC"
    cons = {}
    for n in g.node:
        for i in n.input:
            cons.setdefault(i, []).append(n)
    c3 = cons[c2.output[0]]
    if len(c3) != 1 or c3[0].op_type != "Conv":
        raise SystemExit(f"SUB_PW3: conv.2 feeds {[x.op_type for x in c3]}, not one Conv")
    c3 = c3[0]
    at3 = {a.name: helper.get_attribute_value(a) for a in c3.attribute}
    W3 = numpy_helper.to_array(inits[c3.input[1]]).astype("float32")
    if W3.shape[2:] != (1, 1) or at3.get("group", 1) != 1 or len(c3.input) < 3:
        raise SystemExit(f"SUB_PW3: conv.3 is not a biased 1x1 conv: {W3.shape} {at3}")
    r3 = cons[c3.output[0]]
    if len(r3) != 1 or r3[0].op_type != "Relu":
        raise SystemExit("SUB_PW3: conv.3 is not followed by one ReLU")
    w["w3"] = W3.reshape(-1)
    w["b3"] = numpy_helper.to_array(inits[c3.input[2]]).astype("float32").reshape(-1)
    end, gone = r3[0], {id(c3), id(r3[0])}
out_name = end.output[0] + "_nhwc" if NHWC else end.output[0]
node = helper.make_node("SubConv02", [c0.input[0]], [out_name], name="/pre_encode/subconv02",
                        **{k: v.tolist() for k, v in w.items()}, nhwc=int(NHWC), plugin_version="1", plugin_namespace="")
extra = [helper.make_node("Transpose", [out_name], [end.output[0]], perm=[0, 3, 1, 2], name="/pre_encode/to_nchw")] \
    if NHWC else []
nodes = []
for n in g.node:
    if n is end:
        nodes.append(node)
        nodes.extend(extra)
    elif n is not c0 and n is not relu and n is not c2 and id(n) not in gone:
        nodes.append(n)
del g.node[:]
g.node.extend(nodes)
onnx.save(m, dst, save_as_external_data=True, location=os.path.basename(dst) + ".data")
print("replaced conv.0/ReLU/conv.2" + ("/conv.3/ReLU" if PW3 else "") + " ->", dst, flush=True)
