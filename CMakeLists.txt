# Monorepo build until the split (refactor plan 2026-11-15): each part has its own CMakeLists.txt and also builds alone
# against an installed gen9blas (find_package):
#   lib/           libgen9blas (+ asm/ kernel generators)           -> repo gen9-gpu-blas
#   ggml-backend/  libggml-gen9.so (needs a ggml build)              -> repo ggml-intel-igpu
#   dec emb ina mel silero                                           -> repo the private applications repository
#   cmake -S . -B build -G Ninja && cmake --build build && ctest --test-dir build -L gpu
cmake_minimum_required(VERSION 3.21)
project(gen9-gpu-blas VERSION 0.2.0 LANGUAGES C CXX)

option(GEN9_BUILD_BACKEND "ggml backend (needs GGML_SOURCE_DIR with a CPU-only shared build)" ON)
option(GEN9_BUILD_APPS "applications on top of the library (dec emb ina mel silero)" ON)

add_subdirectory(lib)
if(GEN9_BUILD_BACKEND)
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
