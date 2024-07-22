# 🤩 C++ Applications

{% hint style="info" %}
Currently any source file which includes PhantomFHE's headers must be in `.cu` format to use nvcc as compiler. Pure C++ source files should be converted first.
{% endhint %}

***

There are three common ways to use PhantomFHE in your C++ applications.

## CMake FetchContent

We haven't tested this method, but we would recommend this as prior option because it's more flexible and easy to manage.

Please see [CMake documentation](https://cmake.org/cmake/help/latest/module/FetchContent.html) about this feature for details.

## Git Submodule

Use this git command to add PhantomFHE to your project:

```bash
git submodule add https://github.com/encryptorion-lab/phantom-fhe.git thirdparty/phantom
```

This command will add PhantomFHE as a git submodule in `thirdparty/phantom`. You can customize the path as you want.

## Pre-Installation

See [installation.md](../installation.md "mention") for instructions to install PhantomFHE.

Then you can add the following dependency in your project's `CMakeLists.txt`:

```cmake
find_package(Phantom REQUIRED)
```

You can also customize this step by reading [CMake documentation](https://cmake.org/cmake/help/latest/command/find\_package.html) about `find_package`.
