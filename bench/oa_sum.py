#!/usr/bin/env python3
"""Summary of an i915-perf-reader per-report dump (ComputeBasic counters) for bench/oa.sh.

Usage: oa_sum.py <reader output> [min GpuBusy %, default 95]
Takes the non-idle reports with GpuBusy above the threshold and prints GpuTime-weighted percentages and summed
bytes / summed GPU time for throughputs. "idle" = 100 - EuActive - EuStall: EU time with neither issuing nor a
stalled thread, i.e. no thread on the EU (derived, not a counter). See docs/isa-notes.md, "OA counters".
"""
import sys

PCT = ['EuActive', 'EuStall', 'EuThreadOccupancy', 'EuFpuBothActive', 'Fpu0Active', 'Fpu1Active', 'EuSendActive']
BYTES = ['L3ShaderThroughput', 'GtiReadThroughput', 'GtiWriteThroughput', 'SlmBytesRead', 'SlmBytesWritten']


def main():
    path = sys.argv[1]
    thr = float(sys.argv[2]) if len(sys.argv) > 2 else 95.0
    recs, cur = [], None
    for line in open(path):
        if line.startswith('hw_id='):
            cur = {'idle': 'idle' in line}
            recs.append(cur)
        elif cur is not None and line.startswith('   ') and ':' in line:
            k, v = line.split(':', 1)
            try:
                cur[k.strip()] = float(v)
            except ValueError:
                pass
    busy = [r for r in recs if not r['idle'] and r.get('GpuBusy', 0) > thr and r.get('GpuTime', 0) > 0]
    if not busy:
        sys.exit(f'oa_sum: no report with GpuBusy > {thr} % in {path}')
    t = sum(r['GpuTime'] for r in busy)
    avg = {k: sum(r.get(k, 0) * r['GpuTime'] for r in busy) / t for k in PCT + ['EuAvgIpcRate', 'AvgGpuCoreFrequency']}
    print(f'OA ComputeBasic: {len(busy)} busy reports of {len(recs)}, GPU time {t / 1e6:.1f} ms, '
          f'clock {avg["AvgGpuCoreFrequency"] / 1e6:.0f} MHz')
    idle = 100 - avg['EuActive'] - avg['EuStall']
    both = avg['EuFpuBothActive'] / avg['EuActive'] * 100 if avg['EuActive'] else 0
    print(f'EU active {avg["EuActive"]:.1f} %  stall {avg["EuStall"]:.1f} %  idle {idle:.1f} %  '
          f'occupancy {avg["EuThreadOccupancy"]:.1f} %')
    print(f'FPU both {avg["EuFpuBothActive"]:.1f} % ({both:.0f} % of active)  FPU0 {avg["Fpu0Active"]:.1f} %  '
          f'FPU1 {avg["Fpu1Active"]:.1f} %  send {avg["EuSendActive"]:.1f} %  IPC {avg["EuAvgIpcRate"]:.2f}')
    print('  '.join(f'{k} {sum(r.get(k, 0) for r in busy) / t:.2f} GB/s' for k in BYTES))


if __name__ == '__main__':
    main()
