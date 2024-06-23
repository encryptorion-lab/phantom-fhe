# Python Wrapper for Phantom

## Pre-requisites

- Python environment (Recommend Anaconda)

## First Steps

1. `git submodule update --init` to get submodule pybind11
2. configure Cmake by `cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=<your GPU CC> -DPHANTOM_ENABLE_PYTHON_BINDING=ON`
3. compile Phantom core library by `cmake --build build --target pyPhantom -j` in the project root directory
4. install the library by `sudo cmake --install build`
5. if you are using conda
   1. create new environment for testing: `conda create -n phantom python=3.12`
   2. activate your testing environment by `conda activate phantom`
   3. use `conda develop build/lib` to add `.so` to path (you can unregister this path by `conda develop build/lib --uninstall`)

## Usage

Write your python code to use Phantom library, or try the examples in `python/examples` directory.

## Features

- Support word-wise scheme CKKS

## Common Q&As

1. cannot find Phantom in python: check the python environment is consistent between cmake and python in conda
2. conda error: check whether CMake Python version is the same as the Anaconda environment
## Roadmap

- [x] Support word-wise schemes including BGV, BFV
- [ ] Add more examples
