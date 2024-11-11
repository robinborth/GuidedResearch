# GuidedResearch
We present a novel approach for real-time tracking of parametric head models from monocular RGB-D se- quences. To this end, we propose OptiHead, a fully dif- ferentiable neural optimization pipeline capable of interpo- lating between regression and optimization by adjusting the energy formulation used for second-order solvers, enabling real-time tracking in just a few iterations. Building on pro- jective ICP to establish correspondences, we incorporate neural weighting of residual terms using large receptive fields and regularize the optimization process with a learned prior that accounts for the uncertainty of these weights. Moreover, our method can be trained end-to-end, which was previously not done for neural optimization pipelines for 3DMM, which is achieved by utilizing mesh rasteriza- tion that is capable of forward-mode automatic differenti- ation, allowing efficient and differentiable computation of the Jacobian matrix used during Gauss-Newton optimiza- tion. Our method outperforms ICP for dynamic facial ex- pressions at 21.29 FPS, allowing for real-time applications.


## Installation

```bash
conda create -n guided python=3.10 -y
pip install . -e 
```

# Rasterizer

Make sure that ninja is installed, e.g. this is needed for the torch.utils.cpp_extension to work.

```bash
sudo apt-get install ninja-build
sudo apt-get install libnvidia-gl-535
sudo apt-get install libegl1 
```