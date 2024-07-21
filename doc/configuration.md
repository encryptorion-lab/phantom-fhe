# Configuration

## CMake Options

* `CMAKE_CUDA_ARCHITECTURES`: Set the CUDA architectures to compile for. For example, A100 uses `80`, V100 uses `70`, and P100 uses `60`.
* `PHANTOM_USE_CUDA_PTX`: Enable CUDA PTX optimizations (default: `ON`)
* `PHANTOM_ENABLE_EXAMPLE`: Enable examples (default: `ON`)
* `PHANTOM_ENABLE_BENCH`: Enable benchmarks (default: `ON`)
* `PHANTOM_ENABLE_TEST`: Enable tests (default: `ON`)
* `PHANTOM_ENABLE_PYTHON_BINDING`: Enable Python bindings (default: `ON`)

## Usage

1. Use git to clone this repository recursively (including submodules)
2. Use CMake to configure and build this library
3. Look into build/bin and execute binaries
4. (Optional) Use python bindings (See `python/` directory for details)
