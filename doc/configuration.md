# 🤓 Configuration

## CMake Options

PhantomFHE actually handles the auto-select logic for CUDA architectures. But nvbench still requires to specify it explicitly. If you are using CMake 3.24 and later, you can set it to `native` to let CMake decide.

* `CMAKE_CUDA_ARCHITECTURES` (required by nvbench): Set the CUDA architectures to compile for. For example, A100 uses `80`, V100 uses `70`, and P100 uses `60`. You can also set multiple values with comma as seperator. See [CMake documentation](https://cmake.org/cmake/help/latest/variable/CMAKE\_CUDA\_ARCHITECTURES.html#variable:CMAKE\_CUDA\_ARCHITECTURES) for details.

Phantom has some optional CMake options. If you want to explicitly disable some features, you can add something like `-DPHANTOM_ENABLE_EXAMPLE=OFF` during cmake configuration stage.

* `PHANTOM_USE_CUDA_PTX`: Enable CUDA PTX optimizations (default: `ON`)
* `PHANTOM_ENABLE_EXAMPLE`: Enable examples (default: `ON`)
* `PHANTOM_ENABLE_BENCH`: Enable benchmarks (default: `OFF`)
* `PHANTOM_ENABLE_TEST`: Enable tests (default: `OFF`)
* `PHANTOM_ENABLE_PYTHON_BINDING`: Enable Python bindings (default: `OFF`)
