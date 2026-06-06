- [](index.html)
- Installation

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .md](_sources/installation.md "Download source file")
-  .pdf

# Installation

## Contents

- [Supported platforms](#supported-platforms)
- [CPU](#cpu)
  - [pip installation: CPU](#pip-installation-cpu)
- [NVIDIA GPU](#nvidia-gpu)
  - [pip installation: NVIDIA GPU (CUDA, installed via pip, easier)](#pip-installation-nvidia-gpu-cuda-installed-via-pip-easier)
  - [pip installation: NVIDIA GPU (CUDA, installed locally, harder)](#pip-installation-nvidia-gpu-cuda-installed-locally-harder)
  - [NVIDIA GPU Docker containers](#nvidia-gpu-docker-containers)
- [Google Cloud TPU](#google-cloud-tpu)
  - [pip installation: Google Cloud TPU](#pip-installation-google-cloud-tpu)
- [Mac GPU](#mac-gpu)
- [AMD GPU (Linux)](#amd-gpu-linux)
  - [pip installation: AMD GPU (ROCm)](#pip-installation-amd-gpu-rocm)
- [Intel GPU](#intel-gpu)
- [Conda (community-supported)](#conda-community-supported)
  - [Conda installation](#conda-installation)
- [JAX nightly installation](#jax-nightly-installation)
- [Building JAX from source](#building-jax-from-source)
- [Installing older `jaxlib` wheels](#installing-older-jaxlib-wheels)

# Installation[\#](#installation "Link to this heading")

Using JAX requires installing two packages: `jax`, which is pure Python and cross-platform, and `jaxlib` which contains compiled binaries, and requires different builds for different operating systems and accelerators.

**Summary:** For most users, a typical JAX installation may look something like this:

- **CPU-only (Linux/macOS/Windows)**

      pip install -U jax

- **GPU (NVIDIA, CUDA 13)**

      pip install -U "jax[cuda13]"

- **GPU (AMD, ROCm)**

      pip install -U "jax[rocm7-local]"

- **TPU (Google Cloud TPU VM)**

      pip install -U "jax[tpu]"

## Supported platforms[\#](#supported-platforms "Link to this heading")

The table below shows all supported platforms and installation options. Check if your setup is supported; and if it says *“yes”* or *“experimental”*, then click on the corresponding link to learn how to install JAX in greater detail.

|  | Linux, x86_64 | Linux, aarch64 | Mac, aarch64 | Windows, x86_64 | Windows WSL2, x86_64 |
|----|----|----|----|----|----|
| CPU | [yes](#install-cpu) | [yes](#install-cpu) | [yes](#install-cpu) | [yes](#install-cpu) | [yes](#install-cpu) |
| NVIDIA GPU | [yes](#install-nvidia-gpu) | [yes](#install-nvidia-gpu) | n/a | no | [experimental](#install-nvidia-gpu) |
| Google Cloud TPU | [yes](#install-google-tpu) | n/a | n/a | n/a | n/a |
| AMD GPU | [yes](#install-amd-gpu) | no | n/a | no | [experimental](#install-amd-gpu) |
| Apple GPU | n/a | no | [experimental](#install-mac-gpu) | n/a | n/a |
| Intel GPU | [experimental](#install-intel-gpu) | n/a | n/a | no | no |

## CPU[\#](#cpu "Link to this heading")

### pip installation: CPU[\#](#pip-installation-cpu "Link to this heading")

Currently, the JAX team releases `jaxlib` wheels for the following operating systems and architectures:

- Linux, x86_64

- Linux, aarch64

- macOS, Apple ARM-based

- Windows, x86_64 (*experimental*)

To install a CPU-only version of JAX, which might be useful for doing local development on a laptop, you can run:

    pip install --upgrade pip
    pip install --upgrade jax

On Windows, you may also need to install the [Microsoft Visual Studio 2019 Redistributable](https://learn.microsoft.com/en-US/cpp/windows/latest-supported-vc-redist?view=msvc-170#visual-studio-2015-2017-2019-and-2022) if it is not already installed on your machine.

Other operating systems and architectures require building from source. Trying to pip install on other operating systems and architectures may lead to `jaxlib` not being installed alongside `jax`, although `jax` may successfully install (but fail at runtime).

## NVIDIA GPU[\#](#nvidia-gpu "Link to this heading")

On CUDA 12, JAX supports NVIDIA GPUs that have SM version 5.2 (Maxwell) or newer. Note that Kepler-series GPUs are no longer supported by JAX since NVIDIA has dropped support for Kepler GPUs in its software. On CUDA 13, JAX supports NVIDIA GPUs that have SM version 7.5 or newer. NVIDIA dropped support for previous GPUs in CUDA 13.

You must first install the NVIDIA driver. You’re recommended to install the newest driver available from NVIDIA, but the driver version must be \>= 525 for CUDA 12 on Linux, and \>= 580 for CUDA 13 on Linux.

If you need to use a newer CUDA toolkit with an older driver, for example on a cluster where you cannot update the NVIDIA driver easily, you may be able to use the [CUDA forward compatibility packages](https://docs.nvidia.com/deploy/cuda-compatibility/) that NVIDIA provides for this purpose.

### pip installation: NVIDIA GPU (CUDA, installed via pip, easier)[\#](#pip-installation-nvidia-gpu-cuda-installed-via-pip-easier "Link to this heading")

There are two ways to install JAX with NVIDIA GPU support:

- Using NVIDIA CUDA and cuDNN installed from pip wheels

- Using a self-installed CUDA/cuDNN

The JAX team strongly recommends installing CUDA and cuDNN using the pip wheels, since it is much easier!

NVIDIA has released CUDA packages only for x86_64 and aarch64.

    pip install --upgrade pip

    # NVIDIA CUDA 13 installation
    # Note: wheels only available on linux.
    pip install --upgrade "jax[cuda13]"

    # Alternatively, for CUDA 12, use
    # pip install --upgrade "jax[cuda12]"

We recommend migrating to the CUDA 13 wheels; at some point in the future we will drop CUDA 12 support.

If JAX detects the wrong version of the NVIDIA CUDA libraries, there are several things you need to check:

- Make sure that `LD_LIBRARY_PATH` is not set, since `LD_LIBRARY_PATH` can override the NVIDIA CUDA libraries.

- Make sure that the NVIDIA CUDA libraries installed are those requested by JAX. Rerunning the installation command above should work.

### pip installation: NVIDIA GPU (CUDA, installed locally, harder)[\#](#pip-installation-nvidia-gpu-cuda-installed-locally-harder "Link to this heading")

If you prefer to use a preinstalled copy of NVIDIA CUDA, you must first install NVIDIA [CUDA](https://developer.nvidia.com/cuda-downloads) and [cuDNN](https://developer.nvidia.com/CUDNN).

JAX provides pre-built CUDA-compatible wheels for **Linux x86_64 and Linux aarch64 only**. Other combinations of operating system and architecture are possible, but require building from source (refer to [Building from source](developer.html#building-from-source) to learn more}.

You should use an NVIDIA driver version that is at least as new as your [NVIDIA CUDA toolkit’s corresponding driver version](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions__table-cuda-toolkit-driver-versions). If you need to use a newer CUDA toolkit with an older driver, for example on a cluster where you cannot update the NVIDIA driver easily, you may be able to use the [CUDA forward compatibility packages](https://docs.nvidia.com/deploy/cuda-compatibility/) that NVIDIA provides for this purpose.

JAX currently ships two CUDA wheel variants: CUDA 12 and CUDA 13:

The CUDA 12 wheel is:

| Built with | Compatible with      |
|------------|----------------------|
| CUDA 12.3  | CUDA \>=12.1         |
| CUDNN 9.8  | CUDNN \>=9.8, \<10.0 |
| NCCL 2.19  | NCCL \>=2.18         |

The CUDA 13 wheel is:

| Built with | Compatible with      |
|------------|----------------------|
| CUDA 13.0  | CUDA \>=13.0         |
| CUDNN 9.8  | CUDNN \>=9.8, \<10.0 |
| NCCL 2.19  | NCCL \>=2.18         |

JAX checks the versions of your libraries, and will report an error if they are not sufficiently new. Setting the `JAX_SKIP_CUDA_CONSTRAINTS_CHECK` environment variable will disable the check, but using older versions of CUDA may lead to errors, or incorrect results.

NCCL is an optional dependency, required only if you are performing multi-GPU computations.

To install, run:

    pip install --upgrade pip


    # Installs the wheel compatible with NVIDIA CUDA 13 and cuDNN 9.8 or newer.
    # Note: wheels only available on linux.
    pip install --upgrade "jax[cuda13-local]"

    # Installs the wheel compatible with NVIDIA CUDA 12 and cuDNN 9.8 or newer.
    # Note: wheels only available on linux.
    # pip install --upgrade "jax[cuda12-local]"

**These `pip` installations do not work with Windows, and may fail silently; refer to the table [above](#supported-platforms).**

You can find your CUDA version with the command:

    nvcc --version

JAX uses `LD_LIBRARY_PATH` to find CUDA libraries and `PATH` to find binaries (`ptxas`, `nvlink`). Please make sure that these paths point to the correct CUDA installation.

JAX requires libdevice10.bc, which typically comes from the cuda-nvvm package. Make sure that it is present in your CUDA installation.

Please let the JAX team know on [the GitHub issue tracker](https://github.com/jax-ml/jax/issues) if you run into any errors or problems with the pre-built wheels.

### NVIDIA GPU Docker containers[\#](#nvidia-gpu-docker-containers "Link to this heading")

NVIDIA provides the [JAX Toolbox](https://github.com/NVIDIA/JAX-Toolbox) containers, which are bleeding edge containers containing nightly releases of jax and some models/frameworks.

## Google Cloud TPU[\#](#google-cloud-tpu "Link to this heading")

### pip installation: Google Cloud TPU[\#](#pip-installation-google-cloud-tpu "Link to this heading")

JAX provides pre-built wheels for [Google Cloud TPU](https://cloud.google.com/tpu/docs/users-guide-tpu-vm). To install JAX along with appropriate versions of `jaxlib` and `libtpu`, you can run the following in your cloud TPU VM:

    pip install "jax[tpu]"

For users of Colab (https://colab.research.google.com/), be sure you are using *TPU v2* and not the older, deprecated TPU runtime.

## Mac GPU[\#](#mac-gpu "Link to this heading")

JAX is not supported on Mac/OSX GPU; instead use the standard [CPU installation](#install-cpu) commands.

## AMD GPU (Linux)[\#](#amd-gpu-linux "Link to this heading")

AMD GPU support is provided by a ROCm JAX plugin supported by AMD.

### pip installation: AMD GPU (ROCm)[\#](#pip-installation-amd-gpu-rocm "Link to this heading")

JAX supports a ROCm install via:

    pip install --upgrade "jax[rocm7-local]"

ROCm-specific fixes are shipped as *post-releases* of the ROCm plugin/PJRT packages (for example, `jax-rocm7-plugin==0.9.1.post1`). Upgrading `jax[rocm7-local]` will pick up the newest compatible post-release available from your configured package indexes.

For prerequisites (ROCm/Docker), and for ROCm-specific extras such as `jax[rocm7-local]`, please see [AMD’s Instructions](https://github.com/jax-ml/jax/blob/main/build/rocm/README.md) for details.

**Note**: ROCm support on Windows WSL2 is experimental. For WSL installation, you may need to:

1.  Install [ROCm for WSL](https://rocm.docs.amd.com/projects/install-on-windows/en/latest/tutorial/quick-start.html) following AMD’s official guide

2.  Follow the standard Linux ROCm JAX installation steps within your WSL environment

3.  Be aware that performance and stability may differ from native Linux installations

## Intel GPU[\#](#intel-gpu "Link to this heading")

Intel provides an experimental OneAPI plugin: intel-extension-for-openxla for Intel GPU hardware. For more details and installation instructions, refer to one of the following two methods:

1.  Pip installation: [JAX acceleration on Intel GPU](https://github.com/intel/intel-extension-for-openxla/blob/main/docs/acc_jax.md).

2.  Using [Intel’s XLA Docker container](https://hub.docker.com/r/intel/intel-optimized-xla).

Please report any issues related to:

- JAX: [JAX issue tracker](https://github.com/jax-ml/jax/issues).

- Intel’s OpenXLA plugin: [Intel-extension-for-openxla issue tracker](https://github.com/intel/intel-extension-for-openxla/issues).

## Conda (community-supported)[\#](#conda-community-supported "Link to this heading")

### Conda installation[\#](#conda-installation "Link to this heading")

There is a community-supported Conda build of `jax`. To install it using `conda`, simply run:

    conda install jax -c conda-forge

If you run this command on machine with an NVIDIA GPU, this should install a CUDA-enabled package of `jaxlib`.

To ensure that the jax version you are installing is indeed CUDA-enabled, run:

    conda install "jaxlib=*=*cuda*" jax -c conda-forge

If you would like to override which release of CUDA is used by JAX, or to install the CUDA build on a machine without GPUs, follow the instructions in the [Tips & tricks](https://conda-forge.org/docs/user/tipsandtricks.html#installing-cuda-enabled-packages-like-tensorflow-and-pytorch) section of the `conda-forge` website.

Go to the `conda-forge` [jaxlib](https://github.com/conda-forge/jaxlib-feedstock#installing-jaxlib) and [jax](https://github.com/conda-forge/jax-feedstock#installing-jax) repositories for more details.

## JAX nightly installation[\#](#jax-nightly-installation "Link to this heading")

Nightly releases reflect the state of the main JAX repository at the time they are built, and may not pass the full test suite.

Unlike the instructions for installing a JAX release, here we name all of JAX’s packages explicitly on the command line, so `pip` will upgrade them if a newer version is available.

JAX publishes nightlies, release candidates(RCs), and releases to several non-pypi [PEP 503](https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/) indexes.

All JAX packages can be reached from the index `https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/` as well as PyPI mirrored packages. This additional mirroring enables nightly installation to use –index (-i) as the install method with pip.

**Note:** The unified index could return an RC or release as the newest version even with `--pre` immediately after a release before the newest nightly is rebuilt. If automation or testing must be done against nightlies or you cannot use our full index, use the extra index `https://us-python.pkg.dev/ml-oss-artifacts-published/jax-public-nightly-artifacts-registry/simple/` which only contains nightly artifacts.

The nightly index URLs can also be browsed directly. The `--index` URL is a [PEP 503](https://peps.python.org/pep-0503/) simple repository index for `pip`, and each package has its own sub-directory. For example, you can see the available `jax` packages at [https://us-python.pkg.dev/ml-oss-artifacts-published/jax-public-nightly-artifacts-registry/simple/jax](https://us-python.pkg.dev/ml-oss-artifacts-published/jax-public-nightly-artifacts-registry/simple/jax), `jax-cuda12-pjrt` packages at [https://us-python.pkg.dev/ml-oss-artifacts-published/jax-public-nightly-artifacts-registry/simple/jax-cuda12-pjrt](https://us-python.pkg.dev/ml-oss-artifacts-published/jax-public-nightly-artifacts-registry/simple/jax-cuda12-pjrt), and `jax-cuda13-pjrt` packages at [https://us-python.pkg.dev/ml-oss-artifacts-published/jax-public-nightly-artifacts-registry/simple/jax-cuda13-pjrt](https://us-python.pkg.dev/ml-oss-artifacts-published/jax-public-nightly-artifacts-registry/simple/jax-cuda13-pjrt).

- CPU only:

    pip install -U --pre jax jaxlib -i https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/

- Google Cloud TPU:

    pip install -U --pre jax jaxlib libtpu requests -i https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/ -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

- NVIDIA GPU (CUDA 13):

    pip install -U --pre jax jaxlib "jax-cuda13-plugin[with-cuda]" jax-cuda13-pjrt -i https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/

- NVIDIA GPU (CUDA 12):

    pip install -U --pre jax jaxlib "jax-cuda12-plugin[with-cuda]" jax-cuda12-pjrt -i https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/

## Building JAX from source[\#](#building-jax-from-source "Link to this heading")

Refer to [Building from source](developer.html#building-from-source).

## Installing older `jaxlib` wheels[\#](#installing-older-jaxlib-wheels "Link to this heading")

Due to storage limitations on the Python package index, the JAX team periodically removes older `jaxlib` wheels from the releases on http://pypi.org/project/jax. These can still be installed directly via the URLs here. For example:

    # Install jaxlib on CPU via the wheel archive
    pip install "jax[cpu]==0.3.25" -i https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/

    # Install the jaxlib 0.3.25 CPU wheel directly
    pip install jaxlib==0.3.25 -i https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/

For specific older GPU wheels, be sure to use the `jax_cuda_releases.html` URL; for example

    pip install jaxlib==0.3.25+cuda11.cudnn82 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

[](index.html "previous page")

previous

JAX: High performance array computing

[](notebooks/thinking_in_jax.html "next page")

next

Quickstart: How to think in JAX

Contents

- [Supported platforms](#supported-platforms)
- [CPU](#cpu)
  - [pip installation: CPU](#pip-installation-cpu)
- [NVIDIA GPU](#nvidia-gpu)
  - [pip installation: NVIDIA GPU (CUDA, installed via pip, easier)](#pip-installation-nvidia-gpu-cuda-installed-via-pip-easier)
  - [pip installation: NVIDIA GPU (CUDA, installed locally, harder)](#pip-installation-nvidia-gpu-cuda-installed-locally-harder)
  - [NVIDIA GPU Docker containers](#nvidia-gpu-docker-containers)
- [Google Cloud TPU](#google-cloud-tpu)
  - [pip installation: Google Cloud TPU](#pip-installation-google-cloud-tpu)
- [Mac GPU](#mac-gpu)
- [AMD GPU (Linux)](#amd-gpu-linux)
  - [pip installation: AMD GPU (ROCm)](#pip-installation-amd-gpu-rocm)
- [Intel GPU](#intel-gpu)
- [Conda (community-supported)](#conda-community-supported)
  - [Conda installation](#conda-installation)
- [JAX nightly installation](#jax-nightly-installation)
- [Building JAX from source](#building-jax-from-source)
- [Installing older `jaxlib` wheels](#installing-older-jaxlib-wheels)

By The JAX authors

© Copyright 2024, The JAX Authors.\
