import os

import numpy as np
import tensorrt as trt
from tensorrt_bindings import Logger


def convert_onnx_to_engine(onnx_filename, engine_filename=None):
    logger = Logger(Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, logger)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)

    logger.log(trt.Logger.Severity.INFO, "Parse ONNX file")
    with open(onnx_filename, 'rb') as model:
        if not parser.parse(model.read()):
            logger.log(trt.Logger.ERROR, "ERROR: Failed to parse onnx file")
            for err in range(parser.num_errors):
                logger.log(trt.Logger.ERROR, parser.get_error(err))
            raise RuntimeError("parse onnx file error")

    logger.log(trt.Logger.Severity.INFO, "Building TensorRT engine. This may take a few minutes.")
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_filename:
        with open(engine_filename, 'wb') as f:
            f.write(engine_bytes)


def load_engine(engine_path):
    if not os.path.exists(engine_path):
        raise ValueError(f"onnx file is not exists")
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    return engine_data


def sigmoid(x):
    # 对 x > 0 和 x <= 0 分别进行处理，避免溢出问题
    pos_mask = x >= 0
    neg_mask = x < 0

    # 对于 x >= 0 使用稳定计算公式
    result = np.zeros_like(x)
    result[pos_mask] = 1 / (1 + np.exp(-x[pos_mask]))

    # 对于 x < 0 使用等效的替代公式
    result[neg_mask] = np.exp(x[neg_mask]) / (1 + np.exp(x[neg_mask]))

    return result
