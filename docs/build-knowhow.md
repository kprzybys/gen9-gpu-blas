# Build know-how: optimizing and hardening the C/C++ builds

Research of 2026-11-15, applied to this library, the ggml backend and the applications. Shared settings live in one
CMake module, `lib/cmake/gen9blasBuildOptions.cmake`, installed with the package, so the other projects pick it up
through `find_package(gen9blas)`.

| Topic | Decision here | Why / source |
|---|---|---|
| Default build type | `Release` when none is given (`-O3 -DNDEBUG`); configure prints the build type and the flags, and warns when they are not optimized | A standalone backend build was unoptimized (651 kB instead of 141 kB) because only the library set a default |
| `-O2` vs `-O3` | Measured, see below; default kept as measured | `-O3` adds more optimizations and since GCC 15 drops semantic interposition, but larger code can be slower: MySQL measured 0-6 % in favour of `-O2` ([GCC][gcc-opt], [MySQL][mysql]) |
| Floating point | never `-ffast-math` or `-Ofast` | they change rounding on the host; the golden check requires bit-identical results |
| Symbol visibility | `-fvisibility=hidden` (+ `-fvisibility-inlines-hidden` for C++), explicit exports | smaller, faster shared objects, no interposition of internals; `-fno-semantic-interposition` adds little on top ([SO][interpos]) |
| Reproducible builds | `-ffile-prefix-map=<source>=.` and `-ffile-prefix-map=<build>=build`, `-Werror=date-time`; `SOURCE_DATE_EPOCH` honoured by the tools | `__FILE__` in error messages embedded absolute paths, so the same source built elsewhere had another sha256 (consumers pin by hash) ([reproducible-builds.org][rb], [blog][rbcmake]) |
| LTO | option `GEN9_LTO` (default off), enabled only after `CheckIPOSupported` | CMake policy CMP0069 makes the project check support; host code mostly waits for the GPU, so the gain is small ([CMake][ipo], [CMP0069][cmp0069]) |
| Hardening | option `GEN9_HARDENING` (default on): `-fstack-protector-strong`, `-fstack-clash-protection`, `-fcf-protection`, `-D_FORTIFY_SOURCE=3`, `-Wl,-z,relro,-z,now`, `-Wformat=2 -Wimplicit-fallthrough` | OpenSSF guide; Ubuntu's GCC enables much of this by default, other distributions may not ([OpenSSF][openssf], [Quarkslab][qb]). `-Wconversion` is not enabled yet: it reports many implicit conversions in numeric code, to be fixed one by one with the golden check |
| `-march` | none by default; `GEN9_MARCH` (e.g. `x86-64-v2`, `x86-64-v3`) for builds that never leave one CPU class; never `native` for distributed binaries | `-march=native` binaries may not run on other machines ([GCC x86][gccx86], [MaskRay][maskray], [Gentoo][gentoo]). Note: a llama.cpp built with `GGML_NATIVE=ON` has the same problem in its `libggml-cpu` |
| Build speed | Ninja; ccache used automatically when installed (`GEN9_CCACHE`); generated kernel tables cached by input hash (`GEN9BLAS_KERNEL_CACHE`) | most of a clean build (~2 min) is `ocloc` generating the kernel tables, which ccache cannot speed up; mold, unity builds and precompiled headers do not pay off at this size ([bitsnbites][fastbuild], [IREE][ccache]) |

## Measurement: -O2 vs -O3

See `results/build-o2-o3-2026-11-15.txt`.

[gcc-opt]: https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html
[mysql]: https://dev.mysql.com/worklog/task/?id=10343
[interpos]: https://kaweah.freedombox.rocks/kiwix/content/stackoverflow.com_en_all_2023-11/questions/35745543/new-option-in-gcc-5-3-fno-semantic-interposition
[rb]: https://reproducible-builds.org/docs/build-path/
[rbcmake]: https://solidsmoke.blogspot.com/2021/06/cmake-configuration-for-reproducible.html
[ipo]: https://cmake.org/cmake/help/v4.1/module/CheckIPOSupported.html
[cmp0069]: https://cmake.org/cmake/help/v4.0/policy/CMP0069.html
[openssf]: https://best.openssf.org/Compiler-Hardening-Guides/Compiler-Options-Hardening-Guide-for-C-and-C++
[qb]: https://blog.quarkslab.com/clang-hardening-cheat-sheet-ten-years-later.html
[gccx86]: https://gcc.gnu.org/onlinedocs/gcc/x86-Options.html
[maskray]: https://maskray.me/blog/2022-08-28-march-mcpu-mtune
[gentoo]: https://wiki.gentoo.org/wiki/Safe_CFLAGS
[fastbuild]: https://www.bitsnbites.eu/faster-c-builds/
[ccache]: https://iree.dev/developers/building/cmake-with-ccache/
