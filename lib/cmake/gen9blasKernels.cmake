# gen9blas_kernel_table(<out.c> [SOURCES <file.cl>... OPTIONS <opts|->...] [PROGRAMS <file.bin>...] [PREBUILT <file.c>])
# Kernel table for a program with its own kernels that links gen9blas::kcl. Each .cl in SOURCES is compiled with ocloc
# (-device whl and the matching OPTIONS entry; "-" = no options) into <name>.bin; gen_kernels.py then embeds those and
# the binaries listed in PROGRAMS (full paths, e.g. ${GEN9BLAS_KERNELS_DIR}/gen_s.bin), in this order, into <out.c>
# (tables kcl_programs / kcl_nprograms). Needs ocloc GEN9BLAS_OCLOC_VERSION; without it, PREBUILT names a table
# generated elsewhere (copied as is), or set GEN9BLAS_PREBUILT_TABLES to a directory holding <out name>.
function(gen9blas_kernel_table out)
  cmake_parse_arguments(K "" "PREBUILT" "SOURCES;OPTIONS;PROGRAMS" ${ARGN})
  get_filename_component(stem ${out} NAME_WE)
  get_filename_component(oname ${out} NAME)
  if(NOT K_PREBUILT AND GEN9BLAS_PREBUILT_TABLES AND EXISTS ${GEN9BLAS_PREBUILT_TABLES}/${oname})
    set(K_PREBUILT ${GEN9BLAS_PREBUILT_TABLES}/${oname})
  endif()
  if(K_PREBUILT)
    configure_file(${K_PREBUILT} ${out} COPYONLY)
    return()
  endif()
  list(LENGTH K_SOURCES ns)
  list(LENGTH K_OPTIONS no)
  if(NOT ns EQUAL no)
    message(FATAL_ERROR "gen9blas_kernel_table(${oname}): one OPTIONS entry per source (\"-\" for none)")
  endif()
  find_program(GEN9BLAS_OCLOC ocloc)
  if(NOT GEN9BLAS_OCLOC)
    message(FATAL_ERROR "gen9blas_kernel_table(${oname}): ocloc not found; set GEN9BLAS_PREBUILT_TABLES")
  endif()
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  set(work ${CMAKE_CURRENT_BINARY_DIR}/${stem}.d)
  set(bins)
  set(names)
  set(i 0)
  foreach(s ${K_SOURCES})
    list(GET K_OPTIONS ${i} o)
    math(EXPR i "${i} + 1")
    get_filename_component(n ${s} NAME_WE)
    set(opt)
    if(NOT o STREQUAL "-")
      set(opt -options "${o}")
    endif()
    add_custom_command(OUTPUT ${work}/${n}.bin
      COMMAND ${CMAKE_COMMAND} -E make_directory ${work}
      COMMAND ${GEN9BLAS_OCLOC} compile -q -file ${s} -device whl ${opt} -output ${n} -output_no_suffix -out_dir ${work}
      DEPENDS ${s} VERBATIM)
    list(APPEND bins ${work}/${n}.bin)
    list(APPEND names ${n}.bin)
  endforeach()
  foreach(p ${K_PROGRAMS})
    get_filename_component(n ${p} NAME)
    add_custom_command(OUTPUT ${work}/${n} COMMAND ${CMAKE_COMMAND} -E make_directory ${work}
      COMMAND ${CMAKE_COMMAND} -E copy ${p} ${work}/${n} DEPENDS ${p} VERBATIM)
    list(APPEND bins ${work}/${n})
    list(APPEND names ${n})
  endforeach()
  add_custom_command(OUTPUT ${out}
    COMMAND ${Python3_EXECUTABLE} -I ${GEN9BLAS_GEN_KERNELS} ${work} ${work}/.gen ${names} > ${out}
    DEPENDS ${GEN9BLAS_GEN_KERNELS} ${bins} COMMENT "kernel table ${oname}" VERBATIM)
endfunction()
