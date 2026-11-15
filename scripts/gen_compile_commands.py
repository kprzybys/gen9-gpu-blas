"""Generate compile_commands.json for clangd from `make -n -B` dry runs (nothing is built).
Usage: python3 scripts/gen_compile_commands.py   (from repo root)
Sources not built by any Makefile get the flags of a sibling file in the same dir."""
import json, os, shlex, subprocess, sys
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dirs = sys.argv[1:] or ["lib", "bench", "bench/compare", "asm", "dec", "emb", "ina", "mel", "silero", "ggml-backend", "experiments/no-neo"]
out, seen = [], set()
for d in dirs:
    wd = os.path.join(root, d)
    txt = subprocess.run(['make', '-C', wd, '-n', '-B'], capture_output=True, text=True).stdout.replace('\\\n', ' ')
    for line in txt.splitlines():
        for cmd in line.split('&&'):
            try: a = shlex.split(cmd)
            except ValueError: continue
            if not a or os.path.basename(a[0]) not in ('gcc', 'g++', 'cc', 'c++', 'clang', 'clang++'): continue
            srcs = [x for x in a if x.endswith(('.c', '.cpp', '.cc')) and not x.startswith('-')]
            for s in srcs:
                f = os.path.normpath(os.path.join(wd, s))
                if f in seen or not os.path.exists(f): continue
                seen.add(f)
                args = [x for x in a if x not in srcs or x == s]
                # drop linking/output, add -c
                clean, skip = [], False
                for x in args:
                    if skip: skip = False; continue
                    if x == '-o': skip = True; continue
                    if x.startswith(('-l', '-L', '-Wl,')) or x in ('-shared',): continue
                    clean.append(x)
                clean.insert(1, '-c')
                out.append({'directory': wd, 'file': f, 'arguments': clean})
# fallback: tracked sources not built by any Makefile get flags of a sibling (same dir, same language)
tracked = subprocess.run(['git', '-C', root, 'ls-files', '*.c', '*.cpp'], capture_output=True, text=True).stdout.split()
lib = os.path.join(root, 'lib')
for t in tracked:
    f = os.path.join(root, t)
    if f in seen: continue
    wd = os.path.dirname(f); cpp = f.endswith('.cpp')
    sib = next((e for e in out if e['directory'] == wd and e['file'].endswith('.cpp') == cpp), None) \
       or next((e for e in out if e['file'].endswith('.cpp') == cpp and os.path.dirname(e['file']) == wd), None)
    if sib:
        args = [x for x in sib['arguments'] if x != sib['file'] and not x.endswith(('.c', '.cpp'))] + [f]
    else:
        args = ['g++' if cpp else 'gcc', '-c', '-O2', '-Wall', '-I' + lib, '-I' + os.path.join(lib, 'third_party/mesa'), f]
    out.append({'directory': wd, 'file': f, 'arguments': args}); seen.add(f)
json.dump(out, open(os.path.join(root, 'compile_commands.json'), 'w'), indent=1)
print(len(out), 'entries')
