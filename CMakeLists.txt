cmake_minimum_required(VERSION 3.10)
project(z-vector-search CXX)

set(CMAKE_CXX_STANDARD 17)

if(NOT DEFINED ENV{ZOPEN_PKGINSTALL})
    message(FATAL_ERROR "ZOPEN_PKGINSTALL environment variable not set")
endif()

set(LLAMA_ROOT "$ENV{ZOPEN_PKGINSTALL}/llamacpp/llamacpp")
set(ZOSLIB_ROOT "$ENV{ZOPEN_PKGINSTALL}/zoslib/zoslib")
set(BLIS_ROOT "$ENV{ZOPEN_PKGINSTALL}/blis/blis-2.0")

# z/OS specific flags
set(ZOS_FLAGS "-fzos-le-char-mode=ascii -mnocsect -fno-short-enums -m64 -mzos-target=zosv2r5 -march=z13")
set(ZOS_DEFINES "-D_XOPEN_SOURCE=600 -D_ALL_SOURCE -D_OPEN_SYS_FILE_EXT=1 -D_AE_BIMODAL=1 -D_ENHANCED_ASCII_EXT=0xFFFFFFFF")
set(ZOS_COMP_FLAGS "-fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ZOS_FLAGS} ${ZOS_DEFINES} ${ZOS_COMP_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${ZOS_FLAGS} ${ZOS_DEFINES} ${ZOS_COMP_FLAGS}")

include_directories(${LLAMA_ROOT}/include)

set(LLAMA_LIBS
    ${LLAMA_ROOT}/lib/libllama.a
    ${LLAMA_ROOT}/lib/libggml.a
    ${LLAMA_ROOT}/lib/libggml-base.a
    ${LLAMA_ROOT}/lib/libggml-cpu.a
    ${LLAMA_ROOT}/lib/libggml-blas.a
    ${LLAMA_ROOT}/lib/libmtmd.a
    ${BLIS_ROOT}/lib/libblis.a
    ${ZOSLIB_ROOT}/lib/libzoslib.a
)

add_executable(z-index index.cpp)
target_link_libraries(z-index ${LLAMA_LIBS})

add_executable(z-query query.cpp)
target_link_libraries(z-query ${LLAMA_LIBS})

# Keeping the old one too for compatibility
add_executable(z-vector-search main.cpp)
target_link_libraries(z-vector-search ${LLAMA_LIBS})
