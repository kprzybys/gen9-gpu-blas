"""Generates kernels.c: the ISA code and descriptors of all libgen9blas kernels for the kcl layer (no NEO, directly via i915).

Run (the Makefile does it): python3 gen_kernels.py <bench dir> <work dir> [whl|bdw|bdw-probe|whl-pt] > kernels.c
Programs as in gen9blas.c: OpenCL C sources compiled with `ocloc` with the same options as NEO used to get
(clBuildProgram) and binaries from our assembler (gen_*.bin). For each kernel: SIMD, sizes of cross-thread and
per-thread data, argument offsets (value / stateless pointer / stateful from the binding table), SLM, barrier,
private memory, program constants. Kernels with unsupported features (images, sampler, scratch) get the `unsupported`
flag -- kcl refuses to create them instead of running them with incomplete state.

Targets:
  whl     Gen9.5 (default): zebin, kernel info from .ze_info; tables kcl_programs / kcl_nprograms.
  bdw     Gen8 (Cherryview is compiled as Broadwell, docs/gen8.md): ocloc emits only the patch-token format, read by
          patchtoken_program(); IGC_VISAOptions=-noFP64bRegion; no assembler binaries (they are Gen9 code);
          tables kcl_programs_gen8 / kcl_nprograms_gen8. Kernels whose code contains A64 QWord scattered messages
          (they hang the Cherryview) are marked unsupported.
  bdw-probe  as bdw plus bench/gen8_probe.cl (unmarked: it contains the hanging message on purpose), for
          lib/test_gen8_probe only.
  whl-pt  check of the patch-token reader: Gen9.5 compiled in the patch-token format, compared with whl by
          lib/check_patchtokens.py (assembler binaries are still read from zebin).
"""
import os
import struct
import subprocess
import sys

import yaml

BENCH, WORK = sys.argv[1], sys.argv[2]
TARGET = sys.argv[3] if len(sys.argv) > 3 else "whl"
if TARGET not in ("whl", "bdw", "bdw-probe", "whl-pt"):
    sys.exit(f"gen_kernels: unknown target {TARGET}")
GEN8 = TARGET in ("bdw", "bdw-probe")
DEVICE = "bdw" if GEN8 else "whl"
os.makedirs(WORK, exist_ok=True)

# (file, options) -- exactly like the program_zr / program_bin calls in gen9blas.c
PROGRAMS = [
    ("hgemm.cl", ""), ("sgemm.cl", ""), ("dgemm.cl", ""),
    ("level12.cl", "-cl-std=CL3.0 -DHALF"), ("level12.cl", "-cl-std=CL3.0"), ("level12.cl", "-cl-std=CL3.0 -DDBL"),
    ("gen_h.bin", None), ("gen_hd.bin", None), ("gen_s.bin", None),
]
GEN8_PROBES = [("gen8_probe.cl", "-cl-intel-greater-than-4GB-buffer-required")]   # bring-up probes (bdw-probe only)
# GEN8_PLAIN_PACK: pkf_a/pkf_b in the layout sgemm_v reads; GEN8_CHV: no A64 QWord messages, no indirect DF moves, no
# math.invm/madm (docs/gen8.md 6a)
GEN8_DEFINES = {"sgemm.cl": "-DGEN8_PLAIN_PACK -DGEN8_CHV", "dgemm.cl": "-DGEN8_CHV", "level12.cl": "-DGEN8_CHV"}
VALUES = ("arg_byvalue",)
SPECIAL = {"local_size": "ls", "enqueued_local_size": "els", "global_id_offset": "gid", "global_size": "gs",
             "num_work_groups": "ng"}


def sections(path):
    d = open(path, "rb").read()
    shoff, = struct.unpack_from("<Q", d, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", d, 0x3A)
    secs = [struct.unpack_from("<IIQQQQIIQQ", d, shoff + i * shentsize) for i in range(shnum)]
    stroff = secs[shstrndx][4]
    out = {}
    for s in secs:
        name = d[stroff + s[0]:d.index(b"\0", stroff + s[0])].decode()
        kind, off, size = s[1], s[4], s[5]
        out[name] = (kind, d[off:off + size] if kind != 8 else b"", size)   # 8 = SHT_NOBITS
    return out


def c_bytes(name, b):
    lines = [f"static const uint8_t {name}[{max(len(b), 1)}] = {{"]
    for i in range(0, len(b), 32):
        lines.append("    " + ",".join(str(x) for x in b[i:i + 32]) + ",")
    if not b:
        lines.append("    0,")
    lines.append("};")
    return "\n".join(lines)


def new_kernel(name, isa):
    """Kernel description shared by both readers (what the C descriptor needs)."""
    return {"name": name, "isa": isa, "simd": 0, "end": 0, "pt": 0, "lid": 0, "arg": {},
            "special": {v: [] for v in SPECIAL.values()},
            "other": {"const_base": -1, "private_base_stateless": -1, "work_dimensions": -1},
            "private_size": 0, "slm": 0, "barrier": 0, "bti": {}, "unsup": []}


def zebin_program(binary, dump):
    info = yaml.safe_load(open(os.path.join(dump, ".ze_info")))
    secs = sections(binary)
    constants = b""
    for name in (".data.const", ".bss.const"):
        if name in secs:
            kind, data, size = secs[name]
            constants = data if data else bytes(size)
    kernels = []
    for k in info["kernels"]:
        n, env = k["name"], k["execution_env"]
        text = secs.get(".text." + n)
        if not text:
            continue
        K = new_kernel(n, text[1])
        unsup = K["unsup"]
        # spill_size: the compiler spills registers to scratch memory, which kcl does not configure (MEDIA_VFE_STATE) --
        # such a kernel silently computed wrong results (skinny GEMM MT = 8, 2026-11-12); now an error in kcl_CreateKernel.
        for key in ("per_thread_scratch_size", "spill_size", "has_stack_calls", "inline_data_payload_size"):
            if env.get(key):
                unsup.append(key)
        bti = K["bti"] = {b["arg_index"]: b["bti_value"] for b in k.get("binding_table_indices", [])}
        arg = K["arg"]   # index -> [off, size, ptr, bti, off_addr, off_boff]
        special, other = K["special"], K["other"]
        for a in k.get("payload_arguments", []):
            t = a["arg_type"]
            K["end"] = max(K["end"], a["offset"] + a["size"])
            if t in ("arg_bypointer", "arg_byvalue"):
                e = arg.setdefault(a["arg_index"], [-1, 0, 0, -1, -1, -1])
                if t == "arg_bypointer":
                    e[2] = 1
                    mode, space = a.get("addrmode"), a.get("addrspace")
                    if space not in ("global", "constant"):
                        unsup.append(f"arg{a['arg_index']}:{space}")
                    if mode == "stateful":
                        e[3] = bti[a["arg_index"]]
                    elif mode == "stateless":
                        e[0], e[1] = a["offset"], a["size"]
                    else:
                        unsup.append(f"arg{a['arg_index']}:{mode}")
                else:
                    if e[0] >= 0:   # value in several pieces (structs) -- unsupported
                        unsup.append(f"arg{a['arg_index']}:pieces")
                    e[0], e[1] = a["offset"], a["size"]
            elif t == "buffer_address":
                arg.setdefault(a["arg_index"], [-1, 0, 1, -1, -1, -1])[4] = a["offset"]
            elif t == "buffer_offset":
                arg.setdefault(a["arg_index"], [-1, 0, 1, -1, -1, -1])[5] = a["offset"]
            elif t in SPECIAL:
                special[SPECIAL[t]].append(a["offset"])
            elif t in other:
                other[t] = a["offset"]
            else:
                unsup.append(t)
        for a in k.get("per_thread_payload_arguments", []):
            if a["arg_type"] == "local_id":
                K["lid"] = 1
                K["pt"] += a["size"]
            else:
                unsup.append("per_thread:" + a["arg_type"])
        K["simd"] = env["simd_size"]
        K["private_size"], K["slm"] = env.get("private_size", 0), env.get("slm_size", 0)
        K["barrier"] = 1 if env.get("barrier_count") else 0
        kernels.append(K)
    return constants, kernels


# Patch tokens: definitions in third_party/patchtokens (IGC patch_list.h, patch_shared.h; NEO patchtokens_decoder.cpp,
# kernel_descriptor_from_patchtokens.cpp). All structures are packed, little-endian, u32 fields unless noted.
T_BINDING_TABLE, T_STATEFUL_GLOBAL, T_IMAGE, T_LOCAL_SURFACE, T_DATA_PARAM, T_VFE, T_THREAD_PAYLOAD, T_EXEC_ENV = 8, 11, 12, 15, 17, 18, 22, 23
T_STREAM, T_ARG_INFO, T_ATTRS, T_STRING, T_STATELESS_GLOBAL, T_STATELESS_CONST = 25, 26, 27, 28, 30, 31
T_PRIVATE, T_CONST_SURFACE, T_GLOBAL_SURFACE, T_VFE1 = 38, 44, 43, 55
T_PROG_GLOBAL_DATA, T_PROG_CONST_DATA = 41, 42
IGNORED = {19, 21, T_STREAM, T_ARG_INFO, T_ATTRS, T_STRING, 51, 52}   # IDD/MIDL templates, sizes, names, GTPin
DP_ARG, DP_LWS, DP_GWS, DP_NWG, DP_WDIM, DP_GOFF, DP_ELWS, DP_BOFF, DP_STATEFUL = 1, 2, 3, 4, 5, 16, 28, 42, 43
DP_SPECIAL = {DP_LWS: "ls", DP_ELWS: "els", DP_GOFF: "gid", DP_GWS: "gs", DP_NWG: "ng"}


def tokens(blob):
    pos = 0
    while pos < len(blob):
        tok, size = struct.unpack_from("<II", blob, pos)
        if size < 8 or pos + size > len(blob):
            sys.exit(f"gen_kernels: broken patch token {tok} (size {size})")
        yield tok, blob[pos + 8:pos + size], blob[pos + size:]
        pos += size
        if tok in (T_PROG_GLOBAL_DATA, T_PROG_CONST_DATA):   # followed by InlineDataSize bytes of data
            pos += struct.unpack_from("<I", blob, pos - size + 12)[0]


def patchtoken_program(binary):
    secs = sections(binary)
    d = next(data for kind, data, _ in secs.values() if kind == 0xff000005)   # "Intel(R) OpenCL Device Binary"
    magic, version, _dev, ptr_size, nkernels, _step, plist = struct.unpack_from("<7I", d, 0)
    if magic != 0x494E5443 or version != 1082 or ptr_size != 8:
        sys.exit(f"gen_kernels: {binary}: unexpected patch-token header {magic:#x} v{version} ptr{ptr_size}")
    constants = b""
    for tok, body, rest in tokens(d[28:28 + plist]):
        if tok == T_PROG_CONST_DATA:
            index, size = struct.unpack_from("<II", body, 0)
            if index != 0 or constants:
                sys.exit(f"gen_kernels: {binary}: more than one constant buffer")
            constants = rest[:size]
        elif tok == T_PROG_GLOBAL_DATA:
            sys.exit(f"gen_kernels: {binary}: program-scope global data is not supported")
    pos, kernels = 28 + plist, []
    for _ in range(nkernels):
        _cs, _hash, name_len, plen, isa_len, gsh, dsh, ssh, _unpadded = struct.unpack_from("<IQ7I", d, pos)
        pos += 40
        name = d[pos:pos + name_len].split(b"\0")[0].decode()
        pos += name_len
        isa = d[pos:pos + isa_len]
        pos += isa_len + gsh + dsh
        ssh_blob = d[pos:pos + ssh]
        pos += ssh
        kernels.append(patchtoken_kernel(name, isa, ssh_blob, d[pos:pos + plen]))
        pos += plen
    return constants, kernels


def patchtoken_kernel(name, isa, ssh, plist):
    K = new_kernel(name, isa)
    arg, special, other, unsup = K["arg"], K["special"], K["other"], K["unsup"]
    toks = [(t, b) for t, b, _ in tokens(plist)]
    bt_off = bt_count = None
    for t, b in toks:
        if t == T_BINDING_TABLE:
            bt_off, bt_count, _ = struct.unpack_from("<3I", b)
    table = list(struct.unpack_from(f"<{bt_count}I", ssh, bt_off)) if bt_count else []

    def bti_of(ss_off):   # binding-table index of the surface state at this SSH offset
        if ss_off not in table:
            sys.exit(f"gen_kernels: {name}: surface state {ss_off} not in the binding table")
        return table.index(ss_off)

    def use(off, size):
        K["end"] = max(K["end"], off + size)

    lws_parts = {}   # (kind, instance offset) -> dimensions seen
    stateless_ss, promoted = {}, set()
    for t, b in toks:
        if t == T_EXEC_ENV:
            K["simd"] = struct.unpack_from("<I", b, 12)[0]          # LargestCompiledSIMDSize
            K["barrier"] = 1 if struct.unpack_from("<I", b, 20)[0] else 0   # HasBarriers
            if struct.unpack_from("<I", b, 120)[0]:                 # HasStackCalls
                unsup.append("has_stack_calls")
        elif t == T_THREAD_PAYLOAD:
            f = struct.unpack_from("<20I", b)
            header, lx, ly, lz, flat, _ind, _unused, _glid, _ggid, _goff, _sgo, _sgs, _skip, _ffid, inline = f[:15]
            if header or flat or inline:
                unsup.append(f"thread_payload:header{header}/flat{flat}/inline{inline}")
            if lx or ly or lz:
                K["lid"] = 1   # X, Y, Z rows as written by kcl; K["pt"] is set below, when SIMD is known
                if (lx, ly, lz) != (1, 1, 1):
                    unsup.append(f"local_id_channels:{lx}{ly}{lz}")
        elif t in (T_VFE, T_VFE1):
            if struct.unpack_from("<I", b, 4)[0]:                   # PerThreadScratchSpace
                unsup.append("per_thread_scratch_size")
        elif t == T_LOCAL_SURFACE:
            K["slm"] = struct.unpack_from("<I", b, 4)[0]             # TotalInlineLocalMemorySize
        elif t == T_PRIVATE:
            _ss, off, size, per_thread, simt = struct.unpack_from("<5I", b)
            other["private_base_stateless"] = off
            use(off, size)
            K["private_size"] = per_thread                          # per work-item (IsSimtThread) like zebin
            if not simt:
                unsup.append("private:not_simt")
        elif t == T_CONST_SURFACE:
            _index, _ss, off, size = struct.unpack_from("<4I", b)
            other["const_base"] = off
            use(off, size)
        elif t == T_GLOBAL_SURFACE:
            unsup.append("global_surface")
        elif t == T_IMAGE:
            n, _type, ss_off = struct.unpack_from("<3I", b)
            arg.setdefault(n, [-1, 0, 1, -1, -1, -1])[3] = K["bti"][n] = bti_of(ss_off)
            unsup.append(f"arg{n}:image")
        elif t in (T_STATELESS_GLOBAL, T_STATELESS_CONST):
            n, ss_off, off, size = struct.unpack_from("<4I", b)
            e = arg.setdefault(n, [-1, 0, 0, -1, -1, -1])
            e[0], e[1], e[2] = off, size, 1
            use(off, size)
            stateless_ss[n] = ss_off
        elif t == T_STATEFUL_GLOBAL:
            n, ss_off = struct.unpack_from("<2I", b)
            e = arg.setdefault(n, [-1, 0, 0, -1, -1, -1])
            e[2], e[3] = 1, bti_of(ss_off)
            K["bti"][n] = e[3]
        elif t == T_DATA_PARAM:
            kind, n, off, size, src = struct.unpack_from("<5I", b)
            use(off, size)
            if kind == DP_ARG:
                e = arg.setdefault(n, [-1, 0, 0, -1, -1, -1])
                if e[0] >= 0 and src == e[1] and off == e[0] + e[1]:   # next 4-byte piece of one value (e.g. long)
                    e[1] += size
                    continue
                if e[0] >= 0 or src:   # value in several scattered pieces (structs) -- unsupported
                    unsup.append(f"arg{n}:pieces")
                e[0], e[1] = off, size
            elif kind in DP_SPECIAL:
                dim = src // 4
                lws_parts.setdefault((DP_SPECIAL[kind], off - 4 * dim), set()).add(dim)
            elif kind == DP_WDIM:
                other["work_dimensions"] = off
            elif kind == DP_BOFF:
                arg.setdefault(n, [-1, 0, 1, -1, -1, -1])[5] = off
            elif kind == DP_STATEFUL:
                promoted.add(n)   # pointer argument promoted to stateful: surface + its address in the payload
            else:
                unsup.append(f"data_parameter:{kind}")
        elif t not in IGNORED and t != T_BINDING_TABLE:
            unsup.append(f"patch_token:{t}")
    if K["lid"]:
        K["pt"] = 3 * (64 if K["simd"] == 32 else 32)
    for n in promoted:   # like zebin: stateful (binding table index) + buffer_address, no stateless value
        e = arg[n]
        e[3], e[4], e[0], e[1] = bti_of(stateless_ss[n]), e[0], -1, 0
        K["bti"][n] = e[3]
    for (kind, off), dims in sorted(lws_parts.items(), key=lambda x: x[0][1]):
        special[kind].append(off)
        if dims != {0, 1, 2}:   # kcl writes all three components
            use(off, 12)
    return K


def mark_qword(binary, dump, kernels):
    """Gen8: an A64 QWord scattered message hangs the Cherryview (docs/gen8.md 6a) -> such kernels are unsupported."""
    subprocess.run(["rm", "-rf", dump])
    subprocess.run(["ocloc", "disasm", "-file", binary, "-device", "bdw", "-dump", dump], capture_output=True, check=True)
    for K in kernels:
        asm = open(os.path.join(dump, K["name"] + "_KernelHeap.asm")).read()
        if "qword gathering read" in asm or "qword scattering write" in asm:
            K["unsup"].append("a64_qword_message")


def emit_program(base, constants, kernels):
    print(c_bytes(f"{base}_const", constants))
    descs = []
    for ki, K in enumerate(kernels):
        arg, unsup = K["arg"], list(K["unsup"])
        nargs = max(arg) + 1 if arg else 0
        for i in range(nargs):
            arg.setdefault(i, [-1, 0, 0, -1, -1, -1])
        if nargs > 24:
            unsup.append("over 24 arguments")
        print(c_bytes(f"{base}_k{ki}_isa", K["isa"]))
        def three(xs):
            return "{" + ",".join(str(x) for x in (xs + [-1, -1, -1])[:3]) + "}"
        special, other, bti = K["special"], K["other"], K["bti"]
        args_s = ",".join("{" + ",".join(str(x) for x in arg[i]) + "}" for i in range(nargs)) or "{-1,0,0,-1,-1,-1}"
        descs.append(f'{{"{K["name"]}", {base}_k{ki}_isa, {len(K["isa"])}, {K["simd"]}, {(K["end"] + 31) // 32 * 32}, '
                     f'{K["pt"]}, {K["lid"]}, {nargs}, {{{args_s}}}, {three(special["ls"])}, {three(special["els"])}, '
                     f'{three(special["gid"])}, {three(special["gs"])}, {three(special["ng"])}, {other["work_dimensions"]}, '
                     f'{other["const_base"]}, {other["private_base_stateless"]}, {K["private_size"]}, {K["slm"]}, '
                     f'{K["barrier"]}, {max(bti.values()) + 1 if bti else 0}, "{" ".join(unsup)}"}}')
    print(f"static const struct kcl_kernel_desc {base}_kernels[] = {{\n    " + ",\n    ".join(descs) + "\n};")
    return len(descs)


print("// Generated by lib/gen_kernels.py -- do not edit.")
print('#include "kcl_desc.h"\n')
progs = []
env = dict(os.environ, IGC_VISAOptions="-noFP64bRegion") if GEN8 else None
for pi, (file, options) in enumerate(PROGRAMS + (GEN8_PROBES if TARGET == "bdw-probe" else [])):
    base = f"p{pi}"
    if options is None:
        if GEN8:
            continue   # assembler binaries are Gen9 code
        binary = os.path.join(BENCH, file)
    else:
        binary = os.path.join(WORK, base + ".bin")
        cmd = ["ocloc", "compile", "-file", os.path.join(BENCH, file), "-device", DEVICE, "-output_no_suffix",
               "-output", base, "-out_dir", WORK]
        # Gen8: extra defines for the compile only (the program keeps its recorded options, which gen9blas asks for)
        build_options = (options + " " + GEN8_DEFINES.get(file, "")).strip() if GEN8 else options
        if build_options:
            cmd += ["-options", build_options]
        if TARGET == "whl-pt":
            cmd += ["-internal_options", "-cl-intel-disable-zebin"]
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if r.returncode:
            sys.exit(f"gen_kernels: ocloc {file} {options}: {r.stdout}{r.stderr}")
    if TARGET != "whl" and options is not None:
        constants, kernels = patchtoken_program(binary)
        if GEN8 and (file, options) not in GEN8_PROBES:
            mark_qword(binary, os.path.join(WORK, base + ".d"), kernels)
    else:
        dump = os.path.join(WORK, base + ".d")
        subprocess.run(["rm", "-rf", dump])
        subprocess.run(["ocloc", "disasm", "-file", binary, "-device", "whl", "-dump", dump], capture_output=True)
        constants, kernels = zebin_program(binary, dump)
    n = emit_program(base, constants, kernels)
    progs.append(f'{{"{file}", {"NULL" if options is None else chr(34) + options + chr(34)}, {base}_kernels, '
                 f'{n}, {base}_const, {len(constants)}}}')
suffix = "_gen8" if GEN8 else ""
print(f"const struct kcl_program_desc kcl_programs{suffix}[] = {{\n    " + ",\n    ".join(progs) + "\n};")
print(f"const int kcl_nprograms{suffix} = {len(progs)};")
