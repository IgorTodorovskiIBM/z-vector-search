cmake_minimum_required(VERSION 3.10)
project(z-vector-search C CXX)

set(CMAKE_CXX_STANDARD 17)

# Default to an optimized build — a plain 'cmake ..' otherwise compiles
# sqlite, sqlite-vec, and all tools at -O0.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Dependency Paths (Override with -DLLAMA_ROOT=... etc)
if(DEFINED ENV{ZOPEN_PKGINSTALL})
    set(DEFAULT_LLAMA "$ENV{ZOPEN_PKGINSTALL}/llamacpp/llamacpp")
    set(DEFAULT_ZOSLIB "$ENV{ZOPEN_PKGINSTALL}/zoslib/zoslib")
    set(DEFAULT_BLIS "$ENV{ZOPEN_PKGINSTALL}/blis/blis-2.0")
else()
    set(DEFAULT_LLAMA "/usr/local/llamacpp")
    set(DEFAULT_ZOSLIB "/usr/local/zoslib")
    set(DEFAULT_BLIS "/usr/local/blis")
endif()

set(LLAMA_ROOT "${DEFAULT_LLAMA}" CACHE PATH "Path to llamacpp installation")
# zopen bundles ggml inside the llamacpp package; Homebrew ships it as a
# separate formula. GGML_ROOT defaults to LLAMA_ROOT so zopen/z/OS builds are
# unchanged; macOS dev builds pass -DGGML_ROOT=/opt/homebrew/opt/ggml.
set(GGML_ROOT "${LLAMA_ROOT}" CACHE PATH "Path to ggml installation (defaults to LLAMA_ROOT)")

# Homebrew (macOS and Linux) ships ggml as its own keg, separate from
# llama.cpp — the GGML_ROOT = LLAMA_ROOT default has neither its headers nor
# its libraries. When that default holds and no ggml library sits under it,
# resolve the conventional keg locations up front so the header search and
# the library search below both use the real prefix. Without this a brew
# build configured with only a warning and then failed at link time
# ("libggml.so.0: DSO missing from command line" — GNU ld won't resolve
# symbols from a transitive DT_NEEDED of libllama). On z/OS the brew paths
# never exist, so the probe is a no-op and zopen layouts are unchanged.
if(GGML_ROOT STREQUAL "${LLAMA_ROOT}"
   AND NOT EXISTS "${GGML_ROOT}/lib/libggml.a"
   AND NOT EXISTS "${GGML_ROOT}/lib/libggml.dylib"
   AND NOT EXISTS "${GGML_ROOT}/lib/libggml.so")
    foreach(P "$ENV{HOMEBREW_PREFIX}/opt/ggml" "/opt/homebrew/opt/ggml"
              "/usr/local/opt/ggml" "/home/linuxbrew/.linuxbrew/opt/ggml")
        if(EXISTS "${P}/lib")
            set(GGML_ROOT "${P}")
            message(STATUS "ggml not under LLAMA_ROOT — using ${GGML_ROOT}")
            break()
        endif()
    endforeach()
endif()
set(ZOSLIB_ROOT "${DEFAULT_ZOSLIB}" CACHE PATH "Path to zoslib installation")
set(BLIS_ROOT "${DEFAULT_BLIS}" CACHE PATH "Path to blis installation")

message(STATUS "Using LLAMA_ROOT: ${LLAMA_ROOT}")
message(STATUS "Using GGML_ROOT: ${GGML_ROOT}")
message(STATUS "Using ZOSLIB_ROOT: ${ZOSLIB_ROOT}")
message(STATUS "Using BLIS_ROOT: ${BLIS_ROOT}")

# z/OS specific flags — only applied when actually building on/for z/OS so the
# project stays buildable on macOS/Linux for development and testing.
if(CMAKE_SYSTEM_NAME MATCHES "OS390|zOS")
    # z15 to match the llamacpp zopen build (which already requires it), and
    # -fzvector so sqlite-vec can use the IBM Z vector distance kernels.
    set(ZOS_FLAGS "-fzos-le-char-mode=ascii -mnocsect -fno-short-enums -m64 -mzos-target=zosv2r5 -march=z15 -fzvector")
    set(ZOS_DEFINES "-D_XOPEN_SOURCE=600 -D_ALL_SOURCE -D_OPEN_SYS_FILE_EXT=1 -D_AE_BIMODAL=1 -D_ENHANCED_ASCII_EXT=0xFFFFFFFF")
    set(ZOS_COMP_FLAGS "-fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc")

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ZOS_FLAGS} ${ZOS_DEFINES} ${ZOS_COMP_FLAGS}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${ZOS_FLAGS} ${ZOS_DEFINES} ${ZOS_COMP_FLAGS}")
endif()

# Flexible Header Search
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${LLAMA_ROOT}/include)
include_directories(${LLAMA_ROOT})
if(NOT GGML_ROOT STREQUAL LLAMA_ROOT)
    include_directories(${GGML_ROOT}/include)
endif()
include_directories(${ZOSLIB_ROOT}/include)
include_directories(${BLIS_ROOT}/include)

# SQLite and sqlite-vec (vendored, compiled as C)
add_library(sqlite3 STATIC vendor/sqlite3.c)
target_compile_definitions(sqlite3 PRIVATE
    SQLITE_CORE
    SQLITE_DEFAULT_WAL_SYNCHRONOUS=1
    SQLITE_THREADSAFE=1
    SQLITE_OMIT_LOAD_EXTENSION
)
target_include_directories(sqlite3 PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/vendor>
    $<INSTALL_INTERFACE:include/vectorsearch>
)

add_library(sqlite-vec STATIC vendor/sqlite-vec.c)
target_compile_definitions(sqlite-vec PRIVATE SQLITE_CORE)
if(CMAKE_SYSTEM_NAME MATCHES "OS390|zOS")
    target_compile_definitions(sqlite-vec PRIVATE SQLITE_VEC_ENABLE_ZVECTOR)
endif()
target_include_directories(sqlite-vec PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/vendor>
    $<INSTALL_INTERFACE:include/vectorsearch>
)
target_link_libraries(sqlite-vec sqlite3)

# Header-only public API — consumers link this and get headers +
# sqlite + sqlite-vec. llama.cpp is attached via the installed Config
# file at find_package() time (see cmake/vectorsearchConfig.cmake.in).
add_library(vectorsearch INTERFACE)
target_include_directories(vectorsearch INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include/vectorsearch>
)
target_link_libraries(vectorsearch INTERFACE sqlite3 sqlite-vec)

# Helper function to find a library — prefers static .a (zopen layout),
# falls back to dynamic (Homebrew/Linux) for local development.
function(find_zopen_lib VAR NAME ROOT)
    set(SEARCH_PATHS "${ROOT}/lib" "${ROOT}")
    find_file(${VAR}_FILE
        NAMES "lib${NAME}.a" "lib${NAME}.dylib" "lib${NAME}.so"
        PATHS ${SEARCH_PATHS} NO_DEFAULT_PATH)
    if(${VAR}_FILE)
        set(${VAR} "${${VAR}_FILE}" PARENT_SCOPE)
    else()
        set(${VAR} "${VAR}-NOTFOUND" PARENT_SCOPE)
    endif()
endfunction()

find_zopen_lib(LIB_LLAMA llama ${LLAMA_ROOT})
find_zopen_lib(LIB_GGML ggml ${GGML_ROOT})
find_zopen_lib(LIB_GGML_BASE ggml-base ${GGML_ROOT})
find_zopen_lib(LIB_GGML_CPU ggml-cpu ${GGML_ROOT})
find_zopen_lib(LIB_GGML_BLAS ggml-blas ${GGML_ROOT})
find_zopen_lib(LIB_MTMD mtmd ${LLAMA_ROOT})
find_zopen_lib(LIB_BLIS blis ${BLIS_ROOT})
find_zopen_lib(LIB_ZOSLIB zoslib ${ZOSLIB_ROOT})

# llama and ggml are hard requirements: embedder.h calls ggml directly
# (ggml_backend_dev_by_type), so every tool needs both on its link line.
# Fail at configure time with a hint instead of at link time.
foreach(L LIB_LLAMA LIB_GGML)
    if(NOT ${L})
        message(FATAL_ERROR
            "${L} not found. The embedder requires llama.cpp and ggml; "
            "pass -DLLAMA_ROOT=<prefix> and/or -DGGML_ROOT=<prefix> "
            "(Homebrew: -DGGML_ROOT=$(brew --prefix ggml)).")
    endif()
endforeach()

# Check the rest and warn if missing (some older versions might not have all
# ggml-* split libs; blis/zoslib are z/OS-only).
set(LLAMA_LIBS "")
foreach(L LIB_LLAMA LIB_GGML LIB_GGML_BASE LIB_GGML_CPU LIB_GGML_BLAS LIB_MTMD LIB_BLIS LIB_ZOSLIB)
    if(${L})
        list(APPEND LLAMA_LIBS ${${L}})
    else()
        message(WARNING "Optional or missing dependency: ${L}")
    endif()
endforeach()

set(ALL_LIBS sqlite3 sqlite-vec ${LLAMA_LIBS})

add_executable(z-index index.cpp)
target_link_libraries(z-index ${ALL_LIBS})

add_executable(z-query query.cpp)
target_link_libraries(z-query ${ALL_LIBS})

add_executable(z-vector-search main.cpp)
target_link_libraries(z-vector-search ${ALL_LIBS})

add_executable(z-console console.cpp)
target_link_libraries(z-console ${ALL_LIBS})

add_executable(z-ingest-console ingest_console.cpp)
target_link_libraries(z-ingest-console ${ALL_LIBS})

# z-setup only needs sqlite (no llama.cpp)
set(SETUP_LIBS sqlite3 sqlite-vec)
if(LIB_ZOSLIB)
    list(APPEND SETUP_LIBS ${LIB_ZOSLIB})
endif()
add_executable(z-setup setup.cpp)
target_link_libraries(z-setup ${SETUP_LIBS})

# Store-layer self-test (sqlite-only, not installed). Run manually:
#   ./store-selftest <scratch-dir>
add_executable(store-selftest tests/store_selftest.cpp)
target_link_libraries(store-selftest ${SETUP_LIBS})

install(TARGETS z-index z-query z-vector-search z-console z-ingest-console z-setup DESTINATION bin)

# ---- IBM messages DB parts ----
# Install the pre-packed DB split parts so 'z-setup' can find them at
# ../ibm-docs/ relative to bin/z-setup after installation, without needing
# the source tree or internet access for the DB (model still downloads from
# HuggingFace on first run). Only the .xz.part* files are needed — the full
# .db, PDFs, and GGUF are not installed.
file(GLOB IBMDOCS_PARTS "${CMAKE_SOURCE_DIR}/ibm-docs/*.xz.part*")
if(IBMDOCS_PARTS)
    install(FILES ${IBMDOCS_PARTS} DESTINATION ibm-docs)
else()
    message(WARNING "ibm-docs/*.xz.part* not found — 'z-setup' will require --source-dir or network access")
endif()

# ---- Reusable library install ----
# Lets sibling projects (buildsage, etc.) do:
#     find_package(vectorsearch REQUIRED)
#     target_link_libraries(mytool PRIVATE vectorsearch::vectorsearch)
# and get headers + sqlite + sqlite-vec + llama.cpp transitively.

install(TARGETS vectorsearch sqlite3 sqlite-vec
    EXPORT vectorsearchTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    INCLUDES DESTINATION include/vectorsearch
)

install(FILES
    store_sqlite.h
    common_store.h
    hybrid_search.h
    defaults.h
    embedder.h
    msg_filter.h
    console_parse.h
    DESTINATION include/vectorsearch
)

# Vendor headers live in a subdirectory so store_sqlite.h's existing
# #include "vendor/sqlite3.h" resolves correctly against the install tree.
install(FILES
    vendor/sqlite3.h
    vendor/sqlite3ext.h
    vendor/sqlite-vec.h
    DESTINATION include/vectorsearch/vendor
)

install(EXPORT vectorsearchTargets
    FILE vectorsearchTargets.cmake
    NAMESPACE vectorsearch::
    DESTINATION lib/cmake/vectorsearch
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
    "${CMAKE_SOURCE_DIR}/cmake/vectorsearchConfig.cmake.in"
    "${CMAKE_BINARY_DIR}/vectorsearchConfig.cmake"
    INSTALL_DESTINATION lib/cmake/vectorsearch
)
install(FILES "${CMAKE_BINARY_DIR}/vectorsearchConfig.cmake"
    DESTINATION lib/cmake/vectorsearch
)
