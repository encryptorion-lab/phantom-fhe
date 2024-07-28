---
cover: .gitbook/assets/phantom-cover.png
coverY: 0
layout:
  cover:
    visible: true
    size: hero
  title:
    visible: true
  description:
    visible: true
  tableOfContents:
    visible: true
  outline:
    visible: true
  pagination:
    visible: true
---

# 😃 Getting Started

{% hint style="info" %}
This library (PhantomFHE) is developed and maintained for the research purpose, and the authors are not responsible for any potential issue in production.
{% endhint %}

## Welcome to PhantomFHE!

PhantomFHE is a CUDA-Accelerated Fully Homomorphic Encryption library. It is written in CUDA/C++ and provides Python bindings. It implements all operations of BFV, BGV, and CKKS on GPU (excluding bootstrapping by far).

## Why PhantomFHE?

Nowadays there are a variety of applications based on homomorphic encryption to ensure privacy, especially privacy-preserving machine learning (PPML). But the main limitation of these applications is always the performance. PhantomFHE aims to acclerate FHE schemes using GPUs and improve practicality of HE-based applications.

## Features

* Native GPU acceleration (for NVIDIA GPUs)
* Support word-wise schemes including BGV, BFV, and CKKS (without bootstrapping)
* SOTA performance in most operations
* Easy to integrate with applications (PPML, etc.)

## Quick Start

### Prerequisites

#### Hardwares

* NVIDIA GPU
  * Server: V100, A100, etc.
  * Desktop: RTX 30/40 series, etc.
  * Embedded: AGX Xavier/Orin, etc.
* AMD64/AARCH64 CPU
* stable network for git operations

#### Softwares

* Operating system: Linux only (Debian-based, Arch-based, etc.)
* CUDA Toolkit $$\geq$$ 11.0 (recommended)
* CMake $$\geq$$ 3.20
* GCC  9.0 (recommended)
* Python $$\geq$$ 3.7 (required by pybind11)

### Get PhantomFHE

```bash
git clone --recurse-submodules https://github.com/encryptorion-lab/phantom-fhe.git
```

Notice that you should also clone submodules. If you choose to download source zip, please make sure any CMake configuration related to submodules (nvbench, pybind11) must be turned off.

### Build PhantomFHE

Minimal just-work commands are listed below:

```sh
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build -j
```

If the above commands don't work for you, or you want to build specific targets, please read [configuration.md](configuration.md "mention") for more details. For a fast check-in, we disabled test, bench, and Python binding options by default.

### Run PhantomFHE

Try examples by running:

```bash
./build/bin/example_context
```

## License

This project (PhantomFHE) is released under GPLv3 license. See [LICENSE](../LICENSE) for more information.

Some files contain the modified code from [Microsoft SEAL](https://github.com/microsoft/SEAL). These codes are released under MIT License. See [MIT License](https://github.com/microsoft/SEAL/blob/main/LICENSE) for more information.

Some files contain the modified code from [OpenFHE](https://github.com/openfheorg/openfhe-development). These codes are released under BSD 2-Clause License. See [BSD 2-Clause License](https://github.com/openfheorg/openfhe-development/blob/main/LICENSE) for more information.

## Citation

If you use Phantom in your research, please cite the following paper:

Early access TDSC version ([IEEE Xplore](https://ieeexplore.ieee.org/document/10428046)):

```tex
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

```latex
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

```latex
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
