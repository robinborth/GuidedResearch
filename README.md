# GuidedResearch
We present a novel approach for real-time facial tracking and reconstruction of a monocular video sequence using parametric head models.

## Installation

```bash
conda create -n guided python=3.10 -y
pip install . -e 
```

## TODOS

- Specify how to store the FLAME model


# Rasterizer

Make sure that ninja is installed, e.g. this is needed for the torch.utils.cpp_extension to work.

```bash
sudo apt-get install ninja-build
sudo apt-get install libnvidia-gl-535
sudo apt-get install libegl1 
```