# 🧐 Installation

You can install PhantomFHE to system to allow other CMake-based project find it.

```bash
# after building core library
sudo cmake --install build
```

The default path for installation is `/usr/local` defined by CMake. You can change it by CMake command line arguments.

```bash
# after building core library
cmake --install build --prefix "/home/myuser/installdir"
```
