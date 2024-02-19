# Python Wrapper for Phantom

## First Steps

1. `git submodule update --init` to get pybind11
2. compile
3. if you are using conda, you can use `conda develop build/lib` to add `.so` to path

## Features

- Support word-wise scheme CKKS

## Common Q&As

1. cannot find Phantom in python: check the python environment is consistent between cmake and python in conda

## Roadmap

- [] Support word-wise schemes including BGV, BFV
