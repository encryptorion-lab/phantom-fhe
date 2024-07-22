# PhantomFHE: A CUDA-Accelerated Fully Homomorphic Encryption Library

> [!IMPORTANT]  
> This is a research project and is not intended for production use. We are actively working on improving the
> performance and usability of this library. If you have any questions or suggestions, please feel free to open an issue
> or contact us.

> [!WARNING]  
> This project has been tested on Tesla A100 40G/80G, GTX 3080Ti/3090Ti/4090, AGX Xavier. Other GPUs may have
> compatibility issues and may not give correct results.

## Documentation

Please read [https://encryptorion-lab.gitbook.io/phantom-fhe/](https://encryptorion-lab.gitbook.io/phantom-fhe/) for
detailed instructions and explanations.

## Features

* Native GPU acceleration (for NVIDIA GPUs)
* Support word-wise schemes including BGV, BFV, and CKKS (without bootstrapping)
* SOTA performance in most operations
* Easy to integrate with applications (PPML, etc.)

## License

This project (PhantomFHE) is released under GPLv3 license. See [LICENSE](LICENSE) for more information.

Some files contain the modified code from [Microsoft SEAL](https://github.com/microsoft/SEAL). These codes are released
under MIT License. See [MIT License](https://github.com/microsoft/SEAL/blob/main/LICENSE) for more information.

Some files contain the modified code from [OpenFHE](https://github.com/openfheorg/openfhe-development). These codes are
released under BSD 2-Clause License.
See [BSD 2-Clause License](https://github.com/openfheorg/openfhe-development/blob/main/LICENSE) for more information.

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
