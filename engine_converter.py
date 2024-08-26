from utils import convert_onnx_to_engine

onnx_file_path ="matting_product_v1.onnx"
engine_file_path = "engine.trt"

convert_onnx_to_engine(onnx_file_path, engine_file_path)