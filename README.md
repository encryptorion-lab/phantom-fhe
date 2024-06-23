# Phantom: A CUDA-Accelerated Fully Homomorphic Encryption Library

## Prerequisites

* CUDA >= 11.0
* CMake >= 3.20
* GCC >= 11.0

## CMake Options

* `CMAKE_CUDA_ARCHITECTURES`: Set the CUDA architectures to compile for. For example, A100 uses `80`, V100 uses `70`, and P100 uses `60`.
* `PHANTOM_USE_CUDA_PTX`: Enable CUDA PTX optimizations (default: `ON`)
* `PHANTOM_ENABLE_EXAMPLE`: Enable examples (default: `ON`)
* `PHANTOM_ENABLE_BENCH`: Enable benchmarks (default: `ON`)
* `PHANTOM_ENABLE_TEST`: Enable tests (default: `ON`)
* `PHANTOM_ENABLE_PYTHON_BINDING`: Enable Python bindings (default: `ON`)

## Features

* Native GPU acceleration (for NVIDIA GPUs)
* Support word-wise schemes including BGV, BFV, and CKKS (without bootstrapping)
* SOTA performance in most operations
* Easy to integrate with applications (PPML, etc.)

## Usage

1. Use git to clone this repository recursively (including submodules)
2. Use CMake to configure and build this library
3. Look into build/bin and execute binaries
4. (Optional) Use python bindings (See `python/` directory for details)

## Documentation

Work-in-progress, host at [https://encryptorion-lab.gitbook.io/phantom-fhe/](https://encryptorion-lab.gitbook.io/phantom-fhe/) using GitBook.

## License

This project (Phantom) is released under GPLv3 license. See [COPYING](COPYING) for more information.

Some files contain the modified code from [Microsoft SEAL](https://github.com/microsoft/SEAL). These codes are released under MIT License. See [MIT License](MIT_LICENSE) for more information.

## Citation

If you use Phantom in your research, please cite the following paper:

Early access TDSC version ([IEEE Xplore](https://ieeexplore.ieee.org/document/10428046)):

```
@article{10428046,
         author={Yang, Hao and Shen, Shiyu and Dai, Wangchen and Zhou, Lu and Liu, Zhe and Zhao, Yunlei},
         journal={IEEE Transactions on Dependable and Secure Computing}, 
         title={Phantom: A CUDA-Accelerated Word-Wise Homomorphic Encryption Library}, 
         year={2024},
         volume={},
         number={},
         pages={1-12},
         doi={10.1109/TDSC.2024.3363900}
}
```

IACR ePrint version ([Cryptology ePrint Archive](https://ia.cr/2023/049)):

```
@misc{cryptoeprint:2023/049,
      author = {Hao Yang and Shiyu Shen and Wangchen Dai and Lu Zhou and Zhe Liu and Yunlei Zhao},
      title = {Phantom: A CUDA-Accelerated Word-Wise Homomorphic Encryption Library},
      howpublished = {Cryptology ePrint Archive, Paper 2023/049},
      year = {2023},
      doi = {10.1109/TDSC.2024.3363900},
      note = {\url{https://eprint.iacr.org/2023/049}},
      url = {https://eprint.iacr.org/2023/049}
}
```

If you are exploring BFV optimizations, please also cite the following paper:

```
@misc{cryptoeprint:2023/1429,
      author = {Shiyu Shen and Hao Yang and Wangchen Dai and Lu Zhou and Zhe Liu and Yunlei Zhao},
      title = {Leveraging GPU in Homomorphic Encryption: Framework Design and Analysis of BFV Variants},
      howpublished = {Cryptology ePrint Archive, Paper 2023/1429},
      year = {2023},
      note = {\url{https://eprint.iacr.org/2023/1429}},
      url = {https://eprint.iacr.org/2023/1429}
}
```

## Roadmap

We are planning to support the following features in the future:

* [ ] support bit-wise schemes TFHE/FHEW
* [ ] support scheme switching between word-wise schemes and bit-wise schemes
* [ ] support bootstrapping for BFV/BGV/CKKS
