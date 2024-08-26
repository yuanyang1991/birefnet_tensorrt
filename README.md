# BiRefNet TensorRT Inference

## Introduction

This project provides code for performing inference with [BiRefNet](https://github.com/ZhengPeng7/BiRefNet)
using [TensorRT](https://developer.nvidia.com/tensorrt). The aim is to accelerate the inference process by leveraging
the high-performance capabilities of TensorRT.

## Features

- [x] Efficient inference with BiRefNet using TensorRT.
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

### 2. Place TensorRT Engine in the engine Directory

After conversion, move the generated TensorRT engine file into the engine directory:

```commandline
mv path_to_output_engine engine/
```

### 3. Run Inference

Now, you can run inference using the TensorRT engine with the following command:

```commandline
 python .\infer.py --image-path image_path --output-path result.png --output-alpha-path result_alpha.png --engine-path .\engine.trt

```