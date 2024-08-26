# BiRefNet TensorRT Inference

## Introduction

This project provides code for performing inference with [BiRefNet](https://github.com/ZhengPeng7/BiRefNet)
using [TensorRT](https://developer.nvidia.com/tensorrt). The aim is to accelerate the inference process by leveraging
the high-performance capabilities of TensorRT.

BiRefNet's ONNX model inference requires a significant amount of time; the first inference takes about 5 seconds on my
machine. Even subsequent inferences, though faster, are still slower compared to PyTorch. Therefore, TensorRT is used
for inference in this project. After testing, TensorRT not only reduces the required GPU memory but also improves
inference speed compared to PyTorch. On a 4080 Super GPU, TensorRT uses 6GB of memory and achieves inference in just
**0.13** seconds.

## Features

- [x] Efficient inference with BiRefNet using TensorRT.
- [x] foreground estimate
- [ ] Inference using Docker for an isolated and reproducible environment

## Prerequisites

- NVIDIA GPU with CUDA and Cudnn
- Python 3.9

## Installation

```commandline
pip install -r requirements.txt
```

## Usage

### 1. Convert ONNX Model to TensorRT Engine

First, convert your ONNX model to a TensorRT engine using the provided conversion script:

```python
from utils import convert_onnx_to_engine

onnx_file_path = "birefnet.onnx"
engine_file_path = "engine.trt"

convert_onnx_to_engine(onnx_file_path, engine_file_path)
```

### 2. Run Inference

Now, you can run inference using the TensorRT engine with the following command:

```commandline
 python .\infer.py --image-path image_path --output-path result.png --output-alpha-path result_alpha.png --engine-path .\engine.trt

```