cmake_minimum_required(VERSION 3.24)
project(diffeo LANGUAGES CXX)

# Fastor is header-only; don't build its test suite.
set(BUILD_TESTING OFF)
add_subdirectory(external/Fastor)
add_library(diffeo::Fastor ALIAS Fastor)

add_library(diffeo INTERFACE)
add_library(diffeo::diffeo ALIAS diffeo)
target_include_directories(diffeo INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(diffeo INTERFACE cxx_std_20)
target_link_libraries(diffeo INTERFACE Fastor)

# Building diffeo itself: a target compiling each header on its own (not built by default)
# gives compile_commands.json an entry for every header.
if(PROJECT_IS_TOP_LEVEL)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
    file(GLOB_RECURSE DIFFEO_HEADERS CONFIGURE_DEPENDS include/*.hpp)
    target_sources(diffeo INTERFACE FILE_SET HEADERS BASE_DIRS include FILES ${DIFFEO_HEADERS})
    set_target_properties(diffeo PROPERTIES VERIFY_INTERFACE_HEADER_SETS ON)
endif()
