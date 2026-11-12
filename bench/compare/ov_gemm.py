"""GEMM w OpenVINO (GPU) wobec libgen9blas: graf z jednym MatMul C = A·B, n × n.

Uruchom z katalogu repo: python3 bench/compare/ov_gemm.py <f32|f16> <n> [sekundy]
  OV_STALA=1: B jako stała w grafie (wagi sieci) -- OpenVINO wybiera wtedy ścieżkę FullyConnected z innymi kernelami.
  f32: wejścia FP32, INFERENCE_PRECISION_HINT=f32 (bez tego OpenVINO na GPU liczy po cichu w FP16);
  f16: wejścia FP16, liczenie w FP16.
Czas: infer() na żądaniu z tensorami ustawionymi raz (dane w pamięci hosta, OpenVINO samo decyduje o kopiach);
najlepszy z 5 po rozgrzewce albo średnia z [sekundy] (do bench/power.sh). CPU% jak w l3.c. Błąd: 256 elementów wobec double.
"""
import os
import sys
import time

import numpy as np
import openvino as ov
from openvino import opset13 as ops

prec, n = sys.argv[1], int(sys.argv[2])
sek = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
typ = np.float32 if prec == "f32" else np.float16
ovtyp = ov.Type.f32 if prec == "f32" else ov.Type.f16

stala = os.environ.get("OV_STALA") == "1"
rng = np.random.default_rng(1234)
A = rng.uniform(-1, 1, (n, n)).astype(typ)
B = rng.uniform(-1, 1, (n, n)).astype(typ)
a = ops.parameter([n, n], ovtyp, name="A")
b = ops.constant(B) if stala else ops.parameter([n, n], ovtyp, name="B")
model = ov.Model([ops.matmul(a, b, False, False)], [a] if stala else [a, b], "gemm")
core = ov.Core()
cfg = {"INFERENCE_PRECISION_HINT": prec, "PERFORMANCE_HINT": "LATENCY"}
t0 = time.perf_counter()
cm = core.compile_model(model, "GPU", cfg)
t_kompil = time.perf_counter() - t0

req = cm.create_infer_request()
req.set_input_tensor(0, ov.Tensor(A, shared_memory=True))
if not stala:
    req.set_input_tensor(1, ov.Tensor(B, shared_memory=True))

t0 = time.perf_counter()
req.infer()
t_pierwsze = time.perf_counter() - t0
najl, sciana, cpu, k = 1e30, 0.0, 0.0, 0
t0 = time.perf_counter()
while (time.perf_counter() - t0 < sek) if sek > 0 else k < 5:
    c, t = time.process_time(), time.perf_counter()
    req.infer()
    t = time.perf_counter() - t
    cpu += time.process_time() - c
    sciana += t
    najl = min(najl, t)
    k += 1
if sek > 0:
    najl = sciana / k

C = req.get_output_tensor(0).data
r = np.random.default_rng(99)
I, J = r.integers(0, n, 256), r.integers(0, n, 256)
Ad, Bd = A.astype(np.float64), B.astype(np.float64)
ref = np.einsum("ik,ki->i", Ad[I], Bd[:, J])
mag = np.einsum("ik,ki->i", np.abs(Ad[I]), np.abs(Bd[:, J]))
e = np.abs(C[I, J].astype(np.float64) - ref) / mag
print(f"openvino{'+W' if stala else '  '} gemm  {prec[1:] == '32' and 's' or 'h'} n={n:<5d} host  {najl * 1e3:9.2f} ms  {2 * n ** 3 / najl / 1e9:7.1f} GFLOPS"
      f"  CPU {100 * cpu / sciana:4.0f}%  blad sr={e.mean():.1e} max={e.max():.1e}  "
      f"(kompilacja {t_kompil * 1e3:.0f} ms, 1. wywolanie {t_pierwsze * 1e3:.0f} ms, {k} wywolan)")
