# Building MEDYAN on macOS arm64 (Apple Silicon)

The upstream [`simularium/medyan`](https://github.com/simularium/medyan)
repository is the canonical buildable source for the MEDYAN C++ engine.
It's a 2022-era snapshot that targets Linux x86_64 + Intel-Mac, and
several things break on Apple Silicon with modern toolchains. This
file documents the exact patches we applied to produce a working
arm64 binary on macOS 14+ with Apple clang 16 and CMake 4.

If a future upstream commit fixes any of these, drop the corresponding
patch.

## Prereqs

- macOS arm64 (M1/M2/M3/M4)
- Xcode Command Line Tools (`xcode-select --install`) → Apple clang ≥ 16
- Homebrew: `brew install cmake git pkg-config`
- That's it. Vcpkg builds Boost / Eigen / fmt / spdlog / HDF5 / etc.
  itself. **Do not** also `brew install boost eigen3 hdf5` — they'll
  shadow the vcpkg-built versions at link time and confuse the
  toolchain.

## Sequence

```bash
git clone https://github.com/simularium/medyan.git medyan-src
cd medyan-src
# Apply the seven patches listed below.
MEDYAN_NO_GUI=true ./conf.sh    # ≈20–40 min first run (vcpkg deps)
cd build && make -j8            # ≈3–8 min
ls medyan                       # the binary
```

## Patches

### 1. `CMakeLists.txt` — replace `-march=native` on arm64

Apple clang 16 doesn't accept `-march=native`. Around line 63, replace
the unconditional `else()` branch with an arm64-aware split:

```cmake
if(MSVC)
    set(CMAKE_CXX_FLAGS "/arch:AVX2 /MP /EHsc /bigobj")
else()
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
        set(CMAKE_CXX_FLAGS "-Wno-sign-compare -Wno-unused-variable -Wno-reorder -Wno-unused-but-set-variable -Wno-unused-local-typedefs -Wall -ffast-math -fno-finite-math-only -mcpu=apple-m1 -DFORCE_SCALAR")
    else()
        set(CMAKE_CXX_FLAGS "-Wno-sign-compare -Wno-unused-variable -Wno-reorder -Wno-unused-but-set-variable -Wno-unused-local-typedefs -Wall -ffast-math -fno-finite-math-only -mtune=native -march=native")
    endif()
    set(CMAKE_CXX_FLAGS_DEBUG "-g")
    set(CMAKE_CXX_FLAGS_RELEASE "-O2 -funroll-loops -flto")
endif()
```

`-DFORCE_SCALAR` routes `external/umesimd/UMESimd.h` to its scalar
emulation plugin instead of the broken `plugins/arm/` plugin (which
has C++ template-specialization syntax errors that clang 16 rejects).

### 2. `CMakeLists.txt` — exclude AVX-only modules from the build

After the `file(GLOB_RECURSE src_list ...)` line, drop the x86-specific
modules. They use `__m256i` / `_mm256_*` intrinsics directly and
aren't needed for the binary:

```cmake
if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64")
    list(FILTER src_list EXCLUDE REGEX ".*/Util/DistModule/dist_avx_par\\.cpp$")
    list(FILTER src_list EXCLUDE REGEX ".*/Util/DistModule/dist_example\\.cpp$")
    list(FILTER src_list EXCLUDE REGEX ".*/Util/DistModule/dist_bench\\.cpp$")
    list(FILTER src_list EXCLUDE REGEX ".*/Util/DistModule/dist_mod_vars\\.cpp$")
    list(FILTER src_list EXCLUDE REGEX ".*/TESTS/Util/TestDistModule\\.cpp$")
endif()
```

### 3. Vcpkg triplet — silence two clang 16 errors

Edit
`scripts/.build/vcpkg/triplets/community/arm64-osx.cmake` (created
during `conf.sh`'s first run) and append:

```cmake
set(VCPKG_CXX_FLAGS "-Wno-enum-constexpr-conversion -Wno-deprecated-declarations")
set(VCPKG_C_FLAGS "")
```

Boost 1.78's MPL static_casts trip `-Wenum-constexpr-conversion` (made
an error in clang 16) and various headers use `std::unary_function`
(deprecated and now errored in libc++ 17+).

### 4. Vcpkg helpers — pass `-DCMAKE_POLICY_VERSION_MINIMUM=3.5`

CMake 4 dropped support for `cmake_minimum_required(VERSION < 3.5)` —
many old vcpkg ports (eigen3 3.3.9, szip 2.1.1, …) declare exactly
that. Patch in **two** places (vcpkg has both an old and a new
configure helper):

`scripts/.build/vcpkg/scripts/cmake/vcpkg_configure_cmake.cmake` —
inside `vcpkg_list(APPEND arg_OPTIONS …)`:

```cmake
"-DCMAKE_POLICY_VERSION_MINIMUM=3.5"
```

`scripts/.build/vcpkg/installed/arm64-osx/share/vcpkg-cmake/vcpkg_cmake_configure.cmake` —
after the `if(generator STREQUAL "Ninja")` block:

```cmake
list(APPEND arg_OPTIONS "-DCMAKE_POLICY_VERSION_MINIMUM=3.5")
```

### 5. `src/Util/DistModule/dist_simd_utils.h` — guard x86 debug helpers

Wrap the body that uses `__m128i` / `__m256i` / `__m256` in
`#if defined(__x86_64__) || defined(__amd64__) || defined(_M_X64)`,
keeping only the portable `get_simd_size` template for arm64.

### 6. `src/Util/DistModule/dist_simd.h` — arm64 default algorithm

```cpp
inline auto default_simd_algo(){
#if defined(__x86_64__) || defined(__amd64__) || defined(_M_X64)
    return tag_simd<simd_avx,float>();
#else
    return tag_simd<simd_no,float>();   // dist_serial.h has the overload
#endif
}
```

### 7. `src/Chemistry/HybridBindingSearchManager.h` — alias `t_avx` to serial

Inside the `#ifdef SIMDBINDINGSEARCH` block:

```cpp
#if defined(__x86_64__) || defined(__amd64__) || defined(_M_X64)
    static constexpr dist::tag_simd<dist::simd_avx_par,  float>  t_avx_par {};
    static constexpr dist::tag_simd<dist::simd_avx,      float>  t_avx     {};
#else
    static constexpr dist::tag_simd<dist::simd_no, float>  t_avx_par {};
    static constexpr dist::tag_simd<dist::simd_no, float>  t_avx     {};
#endif
    static constexpr dist::tag_simd<dist::simd_no, float>  t_serial {};
```

### 8. `src/utility.cpp` — arm64 `rdtsc()` fallback

`__asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi))` is x86-only.
Replace the body of `rdtsc()` with an arm64-aware version:

```cpp
unsigned long long rdtsc(){
#ifdef COMPILER_MSVC
    return __rdtsc();
#elif defined(__x86_64__) || defined(__amd64__)
    unsigned int lo,hi;
    __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
    return ((unsigned long long)hi << 32) | lo;
#else
    #include <chrono>
    return static_cast<unsigned long long>(
        std::chrono::steady_clock::now().time_since_epoch().count());
#endif
}
```

### 9. `src/TESTS/TestRand.cpp` — `result_type` typedef on hand-made Engine

libc++ 17+ tightened `__libcpp_random_is_valid_urng` to require a
`result_type` typedef matching `min()`/`max()`/`operator()`. Add one
line to the test's `Engine` struct:

```cpp
struct Engine {
    using result_type = uint64_t;
    uint64_t value = 0;
    // ... rest unchanged
};
```

## Verifying the build

```bash
./build/medyan -h       # Should print "MEDYAN v5.4.0" header
                        # and "SIMD instruction set: none"
```

The wrapper's smoke test exercises three back-to-back checkpoint-restart
intervals end-to-end:

```bash
cd /path/to/pbg-medyan
source .venv/bin/activate
MEDYAN_BIN=/path/to/medyan-src/build/medyan python demo/cxx_smoke.py
```

If you get a working binary on Apple Silicon, expect `runtime ≈ 0.2 s`
per 1-second simulated interval at 5 filaments + actin_only chemistry.

## Why these patches stay in your medyan-src tree, not in pbg-medyan

The MEDYAN source is a separate upstream repository. `pbg-medyan` is a
read-only consumer of the resulting binary — we don't fork or vendor
the source. If you upgrade your medyan checkout, you'll need to re-apply
the patches (or, ideally, by then upstream will have fixed them).
