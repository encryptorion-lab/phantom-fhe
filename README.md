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

```
@article{DBLP:journals/tdsc/YangSDZLZ24,
  author       = {Hao Yang and
                  Shiyu Shen and
                  Wangchen Dai and
                  Lu Zhou and
                  Zhe Liu and
                  Yunlei Zhao},
  title        = {Phantom: {A} CUDA-Accelerated Word-Wise Homomorphic Encryption Library},
  journal      = {{IEEE} Trans. Dependable Secur. Comput.},
  volume       = {21},
  number       = {5},
  pages        = {4895--4906},
  year         = {2024},
  url          = {https://doi.org/10.1109/TDSC.2024.3363900},
  doi          = {10.1109/TDSC.2024.3363900},
  timestamp    = {Fri, 20 Sep 2024 14:01:59 +0200},
  biburl       = {https://dblp.org/rec/journals/tdsc/YangSDZLZ24.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

If you are exploring BFV optimizations, please also cite the following paper:

```
@article{PhantomFHE_BFV,
    author={Shen, Shiyu and Yang, Hao and Dai, Wangchen and Zhou, Lu and Liu, Zhe and Zhao, Yunlei},
    journal={IEEE Transactions on Computers},
    title={Leveraging GPU in Homomorphic Encryption: Framework Design and Analysis of BFV Variants},
    year={2024},
    volume={73},
    number={12},
    pages={2817-2829},
    doi={10.1109/TC.2024.3457733},
}
```

## Roadmap

We are planning to support the following features in the future:

* [ ] support bootstrapping for BFV/BGV/CKKS
* [x] support bit-wise schemes FHEW/TFHE (will not be open-sourced)
* [ ] support scheme switching (will not be open-sourced)
