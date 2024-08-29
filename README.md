# BiRefNet TensorRT Inference

## Introduction

This project provides code for performing inference with [BiRefNet](https://github.com/ZhengPeng7/BiRefNet)
using [TensorRT](https://developer.nvidia.com/tensorrt). The aim is to accelerate the inference process by leveraging
the high-performance capabilities of TensorRT.

BiRefNet's ONNX model inference requires
a [significant amount of time](https://github.com/ZhengPeng7/BiRefNet?tab=readme-ov-file#model-zoo); the first inference
takes about 5 seconds on my
machine. Even subsequent inferences, though faster, are still slower compared to PyTorch. Therefore, TensorRT is used
for inference in this project. After testing, TensorRT not only reduces the required GPU memory but also improves
inference speed compared to PyTorch. On a 4080 Super GPU, TensorRT uses 6GB of memory and achieves inference in just
**0.13** seconds.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1r8GkFPyMMO0OkMX6ih5FjZnUCQrl2SHV?usp=sharing)

## Inference Time Comparison

### 1. First Inference Time

|                                        Method                                        | inference time |
|:------------------------------------------------------------------------------------:|:--------------:|
|  [Pytorch](https://drive.google.com/file/d/1_IfUnu8Fpfn-nerB89FzdNXQ7zk6FKxc/view)   |     0.71s      |
| [ONNX](https://drive.google.com/drive/u/0/folders/1kZM55bwsRdS__bdnsXpkmH6QPyza-9-N) |     5.32s      |
|                                       Tensorrt                                       |     0.17s      |

### 2. Average Inference Time (excluding the first)

|                                        Method                                        | inference time |
|:------------------------------------------------------------------------------------:|:--------------:|
|  [Pytorch](https://drive.google.com/file/d/1_IfUnu8Fpfn-nerB89FzdNXQ7zk6FKxc/view)   |     0.15s      |
| [ONNX](https://drive.google.com/drive/u/0/folders/1kZM55bwsRdS__bdnsXpkmH6QPyza-9-N) |     4.43s      |
|                                       Tensorrt                                       |     0.11s      |

> **Note:**
> 1. Both the PyTorch and ONNX models are from the [official BiRefNet GitHub](https://github.com/ZhengPeng7/BiRefNet).
> 3. The TensorRT model was converted using [this](#Usage) project.
> 3. All tests were conducted on a Win10 system with an RTX 4080 Super.
> 4. Refer to [model_compare.py](./model_compare.py) for the conversion code.

## Features

- [x] Efficient inference with BiRefNet using TensorRT
- [x] foreground estimate
- [x] colab example
- [ ] Performance comparison between PyTorch, ONNX, and TensorRT inference
- [ ] Inference using Docker for an isolated and reproducible environment

## Prerequisites

- NVIDIA GPU with CUDA(>=11.X) and Cudnn(>=8.X)
- Python 3.9

## Installation

```commandline
pip install -r requirements.txt
```

## Usage

### 1. download onnx model

First, download onnx model
from [Google Drive](https://drive.google.com/drive/u/0/folders/1kZM55bwsRdS__bdnsXpkmH6QPyza-9-N)

### 2. Convert ONNX Model to TensorRT Engine

second, convert your ONNX model to a TensorRT engine using the provided conversion script:

```python
from utils import convert_onnx_to_engine

onnx_file_path = "birefnet.onnx"
engine_file_path = "engine.trt"

convert_onnx_to_engine(onnx_file_path, engine_file_path)
```

### 3. Run Inference

Now, you can run inference using the TensorRT engine with the following command:

#### 3.1 single infer

```commandline
 python .\infer.py --image-path image_path --output-path result.png --output-alpha-path result_alpha.png --engine-path .\engine.trt

```

#### 3.2 infer for directory

```commandline
python .\infer.py --image-path image_dir --output-path output_dir --output-alpha-path alpha_dir --engine-path .\engine.trt --mode m
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue if you have any suggestions or
find bugs.

## Thanks

1. [Birefnet](https://github.com/ZhengPeng7/BiRefNet)
2. [fast-foreground-estimation](https://github.com/Photoroom/fast-foreground-estimation)