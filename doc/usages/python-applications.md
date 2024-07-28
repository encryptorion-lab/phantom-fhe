# 🥳 Python Applications

## Setting up Environment

To build Python applications based on PhantomFHE, please install phantomFHE to system first. See [installation.md](../installation.md "mention"). Also make sure CMake option `PHANTOM_ENABLE_PYTHON_BINDING` is set to `ON`.

Anaconda as python environment is recommended because conda can use a simple command to find the PyPhantom library.

If you are using Conda, here are the steps:

1. Create new virtual environment for development.
2. Activate new-created environment.
3. Add PyPhantom library to Python library path.

```bash
conda create -n phantom python=3.12
conda activate phantom
conda develop build/lib # assume your CMake build artifacts are here
```

### Testing

Try the examples in `python/examples` directory first.

### Common Q\&A

* <mark style="color:red;">cannot find Phantom in python</mark>: check the python environment is consistent between CMake and python in conda
* <mark style="color:red;">conda error</mark>: check whether CMake Python version is the same as the Anaconda environment



