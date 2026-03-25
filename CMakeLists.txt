cmake_minimum_required(VERSION 3.10)
project(z-vector-search CXX)

set(CMAKE_CXX_STANDARD 17)

# Dependency Paths (Override with -DLLAMA_ROOT=... etc)
if(DEFINED ENV{ZOPEN_PKGINSTALL})
    set(DEFAULT_LLAMA "${ZOPEN_PKGINSTALL}/llamacpp/llamacpp")
    set(DEFAULT_ZOSLIB "${ZOPEN_PKGINSTALL}/zoslib/zoslib")
    set(DEFAULT_BLIS "${ZOPEN_PKGINSTALL}/blis/blis-2.0")
else()
    set(DEFAULT_LLAMA "/usr/local/llamacpp")
    set(DEFAULT_ZOSLIB "/usr/local/zoslib")
    set(DEFAULT_BLIS "/usr/local/blis")
endif()

set(LLAMA_ROOT "${DEFAULT_LLAMA}" CACHE PATH "Path to llamacpp installation")
set(ZOSLIB_ROOT "${DEFAULT_ZOSLIB}" CACHE PATH "Path to zoslib installation")
set(BLIS_ROOT "${DEFAULT_BLIS}" CACHE PATH "Path to blis installation")

message(STATUS "Using LLAMA_ROOT: ${LLAMA_ROOT}")
message(STATUS "Using ZOSLIB_ROOT: ${ZOSLIB_ROOT}")
message(STATUS "Using BLIS_ROOT: ${BLIS_ROOT}")

# z/OS specific flags
set(ZOS_FLAGS "-fzos-le-char-mode=ascii -mnocsect -fno-short-enums -m64 -mzos-target=zosv2r5 -march=z13")
set(ZOS_DEFINES "-D_XOPEN_SOURCE=600 -D_ALL_SOURCE -D_OPEN_SYS_FILE_EXT=1 -D_AE_BIMODAL=1 -D_ENHANCED_ASCII_EXT=0xFFFFFFFF")
set(ZOS_COMP_FLAGS "-fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ZOS_FLAGS} ${ZOS_DEFINES} ${ZOS_COMP_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${ZOS_FLAGS} ${ZOS_DEFINES} ${ZOS_COMP_FLAGS}")

# Flexible Header Search
include_directories(${LLAMA_ROOT}/include)
include_directories(${LLAMA_ROOT})
include_directories(${ZOSLIB_ROOT}/include)
include_directories(${BLIS_ROOT}/include)

# Helper function to find a library by exact filename
function(find_zopen_lib VAR NAME ROOT)
    set(SEARCH_PATHS "${ROOT}/lib" "${ROOT}")
    find_file(${VAR}_FILE NAMES "lib${NAME}.a" PATHS ${SEARCH_PATHS} NO_DEFAULT_PATH)
    if(${VAR}_FILE)
        set(${VAR} "${${VAR}_FILE}" PARENT_SCOPE)
    else()
        set(${VAR} "${VAR}-NOTFOUND" PARENT_SCOPE)
    endif()
endfunction()

find_zopen_lib(LIB_LLAMA llama ${LLAMA_ROOT})
find_zopen_lib(LIB_GGML ggml ${LLAMA_ROOT})
find_zopen_lib(LIB_GGML_BASE ggml-base ${LLAMA_ROOT})
find_zopen_lib(LIB_GGML_CPU ggml-cpu ${LLAMA_ROOT})
find_zopen_lib(LIB_GGML_BLAS ggml-blas ${LLAMA_ROOT})
find_zopen_lib(LIB_MTMD mtmd ${LLAMA_ROOT})
find_zopen_lib(LIB_BLIS blis ${BLIS_ROOT})
find_zopen_lib(LIB_ZOSLIB zoslib ${ZOSLIB_ROOT})

# Check each one and warn if missing (some older versions might not have all ggml-* libs)
set(LLAMA_LIBS "")
foreach(L LIB_LLAMA LIB_GGML LIB_GGML_BASE LIB_GGML_CPU LIB_GGML_BLAS LIB_MTMD LIB_BLIS LIB_ZOSLIB)
    if(${L})
        list(APPEND LLAMA_LIBS ${${L}})
    else()
        message(WARNING "Optional or missing dependency: ${L}")
    endif()
endforeach()

add_executable(z-index index.cpp)
target_link_libraries(z-index ${LLAMA_LIBS})

add_executable(z-query query.cpp)
target_link_libraries(z-query ${LLAMA_LIBS})

add_executable(z-vector-search main.cpp)
target_link_libraries(z-vector-search ${LLAMA_LIBS})

install(TARGETS z-index z-query z-vector-search DESTINATION bin)
