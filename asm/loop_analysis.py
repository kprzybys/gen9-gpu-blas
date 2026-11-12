"""Loop analysis in machine code: instruction mix and bank conflicts (parity of src1/src2 in mad).
Usage: python3 -I loop_analysis.py <file.asm> <start_label>   (end: first 'while'/'jmpi' line to the label)
"""
import re
import sys

t = open(sys.argv[1]).read().splitlines()
lab = sys.argv[2]
s = next(i for i, l in enumerate(t) if l.startswith(lab + ':'))
e = next(i for i in range(s, len(t)) if re.search(r'(while|jmpi)\b.*\b' + lab + r'\b', t[i]))
body = [l for l in t[s + 1:e + 1] if l.strip() and not re.match(r'^L\d+:', l)]
ops, pattern = {}, {}
conf = nmad = 0
for l in body:
    m = re.match(r'^\s*(\([^)]*\))?\s*([a-z.]+)\s+\((\d+)\|', l)
    if not m:
        continue
    op = f"{m.group(2)}({m.group(3)})"
    ops[op] = ops.get(op, 0) + 1
    if m.group(2) == 'mad':
        regs = re.findall(r'-?r(\d+)\.(\d+)(<[^>]*>)?:(?:hf|f)', l)
        if len(regs) == 4:
            nmad += 1
            s1, s2 = regs[2][0], regs[3][0]
            if int(s1) % 2 == int(s2) % 2 and s1 != s2:
                conf += 1
            k = f"src0{regs[1][2]} src1{regs[2][2]} src2{regs[3][2]}"
            pattern[k] = pattern.get(k, 0) + 1
print("instructions in loop:", len(body))
for k, v in sorted(ops.items(), key=lambda x: -x[1]):
    print(f"  {v:4d} {k}")
print(f"mad: {nmad}, with src1/src2 parity conflict: {conf}")
for k, v in pattern.items():
    print("  operand pattern:", v, k)
