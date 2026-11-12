"""GEMM in OpenVINO (GPU) vs libgen9blas: a graph with a single MatMul C = A·B, n × n.

Run from the repo root: python3 bench/compare/ov_gemm.py <f32|f16> <n> [seconds]
  OV_CONST=1: B as a constant in the graph (network weights) -- OpenVINO then picks the FullyConnected path with different kernels.
  f32: FP32 inputs, INFERENCE_PRECISION_HINT=f32 (without it OpenVINO on the GPU silently computes in FP16);
  f16: FP16 inputs, FP16 computation.
Time: infer() on a request with tensors set once (data in host memory, OpenVINO decides about copies itself);
best of 5 after warm-up, or the mean over [seconds] (for bench/power.sh). CPU% as in l3.c. Error: 256 elements against double.
"""
import os
import sys
import time

import numpy as np
import openvino as ov
from openvino import opset13 as ops

prec, n = sys.argv[1], int(sys.argv[2])
secs = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
dtype = np.float32 if prec == "f32" else np.float16
ov_type = ov.Type.f32 if prec == "f32" else ov.Type.f16

const_b = os.environ.get("OV_CONST") == "1"
rng = np.random.default_rng(1234)
A = rng.uniform(-1, 1, (n, n)).astype(dtype)
B = rng.uniform(-1, 1, (n, n)).astype(dtype)
a = ops.parameter([n, n], ov_type, name="A")
b = ops.constant(B) if const_b else ops.parameter([n, n], ov_type, name="B")
model = ov.Model([ops.matmul(a, b, False, False)], [a] if const_b else [a, b], "gemm")
core = ov.Core()
cfg = {"INFERENCE_PRECISION_HINT": prec, "PERFORMANCE_HINT": "LATENCY"}
t0 = time.perf_counter()
cm = core.compile_model(model, "GPU", cfg)
t_compile = time.perf_counter() - t0

req = cm.create_infer_request()
req.set_input_tensor(0, ov.Tensor(A, shared_memory=True))
if not const_b:
    req.set_input_tensor(1, ov.Tensor(B, shared_memory=True))

t0 = time.perf_counter()
req.infer()
t_first = time.perf_counter() - t0
best, wall, cpu, k = 1e30, 0.0, 0.0, 0
t0 = time.perf_counter()
while (time.perf_counter() - t0 < secs) if secs > 0 else k < 5:
    c, t = time.process_time(), time.perf_counter()
    req.infer()
    t = time.perf_counter() - t
    cpu += time.process_time() - c
    wall += t
    best = min(best, t)
    k += 1
if secs > 0:
    best = wall / k

C = req.get_output_tensor(0).data
r = np.random.default_rng(99)
I, J = r.integers(0, n, 256), r.integers(0, n, 256)
Ad, Bd = A.astype(np.float64), B.astype(np.float64)
ref = np.einsum("ik,ki->i", Ad[I], Bd[:, J])
mag = np.einsum("ik,ki->i", np.abs(Ad[I]), np.abs(Bd[:, J]))
e = np.abs(C[I, J].astype(np.float64) - ref) / mag
print(f"openvino{'+W' if const_b else '  '} gemm  {prec[1:] == '32' and 's' or 'h'} n={n:<5d} host  time={best * 1e3:9.2f} ms  {2 * n ** 3 / best / 1e9:7.1f} GFLOPS"
      f"  CPU {100 * cpu / wall:4.0f}%  rel_err: avg={e.mean():.1e} max={e.max():.1e}  "
      f"(compile {t_compile * 1e3:.0f} ms, first call {t_first * 1e3:.0f} ms, {k} calls)")
