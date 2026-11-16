"""Hoist `__local` declarations to kernel scope in TVM-generated OpenCL C (needed on Intel NEO).

TVM's OpenCL codegen declares cross-thread reduction buffers (`__local half red_buf0_ptr[64];`) inside nested
blocks. OpenCL C 1.2/3.0 (s6.5.2) allows `__local` variables only in the outermost scope of a kernel; Adreno's
compiler accepts the nested form, Intel's clang-based frontend (NEO 24.35) rejects it with
"variables in the local address space can only be declared in the outermost scope of a kernel function".

Registered as `tvm_callback_opencl_postproc`, which TVM calls on each kernel's source before building the
module (codegen_opencl.cc). Import this module before compiling (see sitecustomize.py next to it).

Identical declarations from sibling scopes are merged into one (each scope uses the buffer only between its
own barriers); two different declarations under one name raise an error instead of guessing.
"""

import re

import tvm_ffi

_DECL = re.compile(r"^(?P<indent>[ \t]+)(?P<decl>__local\s+[\w\s]+?\b(?P<name>\w+)\s*\[[^\]]+\]\s*;)\s*$")


def hoist_local(source: str) -> str:
    lines = source.split("\n")
    out = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        out.append(line)
        if "__kernel" not in line:
            i += 1
            continue
        # find the opening brace of this kernel's body
        j = i
        while "{" not in lines[j]:
            j += 1
            out.append(lines[j])
        body_open = len(out)  # insert position after the line with "{"
        depth = lines[j].count("{") - lines[j].count("}")
        k = j + 1
        decls = {}
        body = []
        while k < n and depth > 0:
            m = _DECL.match(lines[k])
            if m and depth > 1:
                name = m.group("name")
                decl = " ".join(m.group("decl").split())
                if name in decls and decls[name] != decl:
                    raise ValueError(f"conflicting __local declarations of {name}: {decls[name]} / {decl}")
                decls[name] = decl
            else:
                body.append(lines[k])
            depth += lines[k].count("{") - lines[k].count("}")
            k += 1
        out[body_open:body_open] = ["  " + d for d in decls.values()]
        out.extend(body)
        i = k
    return "\n".join(out)


@tvm_ffi.register_global_func("tvm_callback_opencl_postproc", override=True)
def _postproc(source, target):  # noqa: ARG001
    return hoist_local(str(source))
