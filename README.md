# Real-World Image Super-Resolution Via Spatio-Temporal Correlation Network(STCN)

PyTorch code for our paper "Real-World Image Super-Resolution Via Spatio-Temporal Correlation Network" 

The code is built on RCAN (PyTorch) and tested on Ubuntu 18.04 environment (Python3.6, PyTorch_0.4.0) with Titan X/1080Ti GPUs.


# Introduction

Super-resolving real-world image is very challenging due to the degradations in real-world low-resolution images are highly complicated. In this paper, we propose a novel Spatio-temporal Correlation Network (STCN) for real-world single image super-resolution. Specifically, we adopt a very deep network which consists of several attention groups. Each attention group (AG) contains a series of residual channel attention blocks (RCABs) and one spatio-temporal correlation block (STCB). Notably, STCB mainly consists of a residual 3D convolution, and aims to fully explore the local spatial and temporal correlations between channels of feature maps generated by RCABs for selectively capturing more informative features. In addition, we propose an innovative dual restriction (DR) through a simple degradation model to reduce the possible space of mapping functions in super-resolution. Experiments conducted on two public available real-world datasets demonstrate the superior performance of our method.

The architecture of our proposed Spatio-Temporal Correlation Network(STCN):
<img width="1603" alt="1_1" src="https://user-images.githubusercontent.com/75960553/111739951-d5610080-88be-11eb-99b8-ca59f2336395.png">
Illustration of the RCAB and the STCB architecture.
<img width="1496" alt="2" src="https://user-images.githubusercontent.com/75960553/111739977-e0b42c00-88be-11eb-8301-02b65895b855.png">


