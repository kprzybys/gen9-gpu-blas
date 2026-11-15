# Top-level build. This repository holds the library (lib/, asm/ kernel generators); the ggml backend
# (github.com/kprzybys/ggml-intel-igpu) and the applications are separate repositories that build against an installed
# gen9blas (find_package). When their trees are placed next to lib/ (ggml-backend/, dec emb ina mel silero; e.g. the
# composed workspace used by the regression checks), they are built here too.
#   cmake -S . -B build -G Ninja && cmake --build build && ctest --test-dir build -L gpu
cmake_minimum_required(VERSION 3.21)
project(gen9-gpu-blas VERSION 0.2.0 LANGUAGES C CXX)

option(GEN9_BUILD_BACKEND "ggml backend, when ggml-backend/ is present (needs GGML_SOURCE_DIR with a CPU-only shared build)" ON)
option(GEN9_BUILD_APPS "applications on top of the library, when present (dec emb ina mel silero)" ON)

add_subdirectory(lib)
if(GEN9_BUILD_BACKEND AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/ggml-backend/CMakeLists.txt)
  set(GGML_SOURCE_DIR "$ENV{HOME}/src/llama.cpp" CACHE PATH "llama.cpp or whisper.cpp source tree")
  if(EXISTS ${GGML_SOURCE_DIR}/ggml/include/ggml-backend.h)
    add_subdirectory(ggml-backend)
  else()
    message(WARNING "GEN9_BUILD_BACKEND: no ggml in GGML_SOURCE_DIR=${GGML_SOURCE_DIR}; backend skipped")
  endif()
endif()
if(GEN9_BUILD_APPS)
  foreach(app dec emb ina mel silero)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${app}/CMakeLists.txt)
      add_subdirectory(${app})
    endif()
  endforeach()
endif()
