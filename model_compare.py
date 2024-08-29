import os

import numpy as np
import torch
import time
import onnxruntime as ort
from tensorrt_bindings import Logger
import tensorrt as trt
import common
from common_runtime import load_engine

from models.birefnet import BiRefNet
from utils import check_state_dict


class PerformanceMonitor:
    def __init__(self):
        self.reset()

    def reset(self):
        self.start_time = 0
        self.end_time = 0
        self.elapsed_time = 0

    def start(self):
        torch.cuda.synchronize()  # Ensure all CUDA operations are complete
        self.start_time = time.perf_counter()

    def stop(self):
        torch.cuda.synchronize()  # Ensure all CUDA operations are complete
        self.end_time = time.perf_counter()
        self.elapsed_time = self.end_time - self.start_time

    def get_stats(self):
        return {
            "elapsed_time_s": self.elapsed_time
        }


class BaseModelEvaluator:
    def __init__(self, model, input_tensor):
        self.model = model
        self.input_tensor = input_tensor
        self.monitor = PerformanceMonitor()

    def _run_inference(self):
        raise NotImplementedError("Subclasses should implement this method")

    def measure_first_inference(self):
        self.monitor.reset()
        self.monitor.start()
        _ = self._run_inference()
        self.monitor.stop()
        return self.monitor.get_stats()

    def measure_average_inference(self, num_warmup=3, num_runs=10):
        # Warm-up runs
        for _ in range(num_warmup):
            _ = self._run_inference()

        total_time_taken = 0

        for _ in range(num_runs):
            self.monitor.reset()
            self.monitor.start()
            _ = self._run_inference()
            self.monitor.stop()
            stats = self.monitor.get_stats()
            total_time_taken += stats["elapsed_time_s"]

        avg_time_taken = total_time_taken / num_runs

        return {
            "avg_elapsed_time_s": avg_time_taken
        }


class PyTorchModelEvaluator(BaseModelEvaluator):
    def _run_inference(self):
        with torch.no_grad():
            output = self.model(self.input_tensor)
        return output


class ONNXModelEvaluator(BaseModelEvaluator):

    def __init__(self, model_path, input_tensor):
        self.input_tensor = input_tensor
        self.monitor = PerformanceMonitor()
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.ort_session = ort.InferenceSession(model_path, sess_options=session_options,
                                                providers=['CUDAExecutionProvider'])

    def _run_inference(self):
        output = self.ort_session.run(None, {self.ort_session.get_inputs()[0].name: self.input_tensor.cpu().numpy()})
        return output


class TensorrtEvaluator(BaseModelEvaluator):

    def __init__(self, engine_path, input_tensor):
        logger = Logger(Logger.INFO)
        runtime = trt.Runtime(logger)
        self.monitor = PerformanceMonitor()
        self.engine = runtime.deserialize_cuda_engine(load_engine(engine_path))
        self.context = self.engine.create_execution_context()
        self.image_data = input_tensor.cpu().numpy().ravel()

    def _run_inference(self):
        inputs, outputs, bindings, stream = common.allocate_buffers(self.engine)
        np.copyto(inputs[0].host, self.image_data)
        trt_outputs = common.do_inference(self.context, self.engine, bindings, inputs, outputs, stream)


def clear_pytorch_context():
    """Clears the PyTorch CUDA context to minimize interference with ONNX Runtime"""
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def evaluate_pytorch_models(pytorch_model, input_tensor):
    input_tensor = input_tensor.to('cuda')

    # PyTorch model evaluation
    print("Evaluating PyTorch model:")
    pytorch_evaluator = PyTorchModelEvaluator(pytorch_model, input_tensor)
    first_inference_stats_pt = pytorch_evaluator.measure_first_inference()
    avg_inference_stats_pt = pytorch_evaluator.measure_average_inference()

    print(f"PyTorch first inference time: {first_inference_stats_pt['elapsed_time_s']:.4f} seconds")
    print(f"PyTorch average inference time: {avg_inference_stats_pt['avg_elapsed_time_s']:.4f} seconds")

    # Clear PyTorch context before evaluating ONNX model
    clear_pytorch_context()


def evaluate_onnx_models(onnx_model_path, input_tensor):
    # ONNX model evaluation
    print("\nEvaluating ONNX model:")
    onnx_evaluator = ONNXModelEvaluator(onnx_model_path, input_tensor)
    first_inference_stats_onnx = onnx_evaluator.measure_first_inference()
    avg_inference_stats_onnx = onnx_evaluator.measure_average_inference()

    print(f"ONNX first inference time: {first_inference_stats_onnx['elapsed_time_s']:.4f} seconds")
    print(f"ONNX average inference time: {avg_inference_stats_onnx['avg_elapsed_time_s']:.4f} seconds")


def evaluate_tensorrt_models(engine_path, input_tensor):
    print("\nEvaluating tensorrt model:")
    tensorrt_evaluator = TensorrtEvaluator(engine_path=engine_path, input_tensor=input_tensor)

    first_inference_stats_tensorrt = tensorrt_evaluator.measure_first_inference()
    avg_inference_stats_tensorrt = tensorrt_evaluator.measure_average_inference()

    print(f"tensorrt first inference time: {first_inference_stats_tensorrt['elapsed_time_s']:.4f} seconds")
    print(f"tensorrt average inference time: {avg_inference_stats_tensorrt['avg_elapsed_time_s']:.4f} seconds")


input_tensor = torch.randn(1, 3, 1024, 1024)
pytorch_model = BiRefNet(bb_pretrained=False)
state_dict = torch.load('BiRefNet-general-epoch_244.pth', map_location='cpu')
state_dict = check_state_dict(state_dict)
pytorch_model.load_state_dict(state_dict)
pytorch_model.eval()
pytorch_model.cuda()
torch.set_float32_matmul_precision(['high', 'highest'][0])
onnx_model_path = "output.onnx"

# Execute evaluation
# evaluate_pytorch_models(pytorch_model, input_tensor)
# evaluate_onnx_models(onnx_model_path, input_tensor)
evaluate_tensorrt_models("engine.trt", input_tensor)
