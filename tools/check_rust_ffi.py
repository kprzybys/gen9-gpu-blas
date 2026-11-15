#!/usr/bin/env python3
"""Check bindings/rust/gen9blas-sys/src/lib.rs against lib/gen9blas.h + lib/gen9blas_ml.h.

Every GEN9BLAS_API function must be declared in Rust with the same parameter count and matching types (C type ->
Rust type mapping below), and nothing extra. Also checks the numeric constants that both sides define.
usage: tools/check_rust_ffi.py   (exit 1 on a mismatch)
"""
import os, re, subprocess, sys

ROOT = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True,
                      cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
BASE = {"int": "c_int", "double": "c_double", "float": "c_float", "char": "c_char", "void": "c_void", "size_t": "usize",
        "int64_t": "i64", "uint16_t": "u16", "gen9blas_packed_b": "gen9blas_packed_b",
        "gen9blas_packed_a": "gen9blas_packed_a"}


def c_to_rust(decl, is_param=True):
    """'const float *const *C' -> '*const *mut c_float' style; arrays become pointers."""
    d = re.sub(r"\b\w+\s*\[\s*\d*\s*\]$", lambda m: "*" + m.group(0).split("[")[0], decl.strip())  # x[4] -> *x
    arr = decl.strip().endswith("]")
    d = re.sub(r"\[\s*\d*\s*\]", "", decl.strip())
    if is_param:
        d = re.sub(r"\s*\b[A-Za-z_]\w*$", "", d) if re.search(r"[\w\*]\s+\**\s*[A-Za-z_]\w*$", d) else d
    toks = re.findall(r"const|\*|\w+", d)
    base = next(t for t in toks if t not in ("const", "*", "unsigned", "struct"))
    # walk: [const] base [const] (* [const])*   pointer constness applies to what the pointer points at
    out, pend_const, seen_base = [], False, False
    quals = []  # for each '*': is the pointee const
    pointee_const = False
    for t in toks:
        if t == "const": pend_const = True; continue
        if t == "*": quals.append(pointee_const or pend_const); pointee_const = False; pend_const = False; continue
        if t in ("unsigned", "struct"): continue
        pointee_const = pend_const; pend_const = False
    if arr: quals.append(pointee_const)
    rust = BASE[base]
    for q in quals:  # innermost first
        rust = ("*const " if q else "*mut ") + rust
    return rust


def c_funcs():
    out = {}
    for h in ("lib/gen9blas.h", "lib/gen9blas_ml.h"):
        txt = re.sub(r"//[^\n]*|^\s*#[^\n]*", "", open(os.path.join(ROOT, h)).read(), flags=re.M)
        for m in re.finditer(r"GEN9BLAS_API\s+(.*?)\b(gen9blas_\w+)\s*\((.*?)\)\s*;", txt, re.S):
            ret, name, params = m.group(1).strip(), m.group(2), " ".join(m.group(3).split())
            ps = [] if params in ("", "void") else [c_to_rust(p) for p in params.split(",")]
            r = None if ret == "void" else c_to_rust(ret + " x")
            out[name] = (ps, r)
    return out


def rust_funcs():
    txt = open(os.path.join(ROOT, "bindings/rust/gen9blas-sys/src/lib.rs")).read()
    txt = re.sub(r"//[^\n]*", "", txt)
    out = {}
    for m in re.finditer(r"pub fn (gen9blas_\w+)\s*\((.*?)\)\s*(?:->\s*([^;]+))?;", txt, re.S):
        name, params, ret = m.group(1), " ".join(m.group(2).split()), m.group(3)
        ps = [p.split(":", 1)[1].strip() for p in params.split(",") if p.strip()]
        out[name] = ([" ".join(p.split()) for p in ps], " ".join(ret.split()) if ret else None)
    return out


def consts():
    c = {}
    for h in ("lib/gen9blas.h", "lib/gen9blas_ml.h"):
        txt = re.sub(r"//[^\n]*", "", open(os.path.join(ROOT, h)).read())
        for m in re.finditer(r"#define (GEN9BLAS_VERSION_\w+|GEN9BLAS_KC)\s+(-?\d+)", txt): c[m.group(1)] = int(m.group(2))
        for body in re.findall(r"enum\s*\{(.*?)\}", txt, re.S):
            v = -1
            for item in body.split(","):
                item = item.strip()
                if not item: continue
                mm = re.match(r"(GEN9BLAS_\w+)(?:\s*=\s*(-?\d+))?", item)
                if not mm: continue
                v = int(mm.group(2)) if mm.group(2) is not None else v + 1
                c[mm.group(1)] = v
    r = {}
    for m in re.finditer(r"pub const (GEN9BLAS_\w+): c_int = (-?\d+);",
                         open(os.path.join(ROOT, "bindings/rust/gen9blas-sys/src/lib.rs")).read()):
        r[m.group(1)] = int(m.group(2))
    return c, r


def main():
    c, r = c_funcs(), rust_funcs()
    bad = 0
    for n in sorted(set(c) | set(r)):
        if n not in r: print(f"missing in Rust: {n}"); bad += 1; continue
        if n not in c: print(f"not in the C headers: {n}"); bad += 1; continue
        if c[n] != r[n]:
            print(f"{n}:\n  C    {c[n]}\n  Rust {r[n]}"); bad += 1
    cc, rc = consts()
    for k, v in sorted(rc.items()):
        if k not in cc: print(f"constant {k} not in the C headers"); bad += 1
        elif cc[k] != v: print(f"constant {k}: C {cc[k]}, Rust {v}"); bad += 1
    for k in sorted(set(cc) - set(rc)):
        print(f"constant {k} missing in Rust"); bad += 1
    print(f"{len(c)} functions, {len(cc)} constants: {'OK' if not bad else f'{bad} mismatches'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
