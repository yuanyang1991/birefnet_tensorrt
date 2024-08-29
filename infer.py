import argparse
import os

import cv2
import numpy as np
import tensorrt as trt
from PIL import Image
from tensorrt_bindings import Logger

import common
from transformers import Compose, Resize, ToTensor, Normalize, ToPILImage
from utils import load_engine, sigmoid

logger = Logger(Logger.INFO)


def fb_blur_fusion_foreground_estimator_2(image, alpha, blur_radius=90):
    """
    Estimate the foreground image by applying a blur fusion method.

    Args:
        image (numpy.ndarray): The input image.
        alpha (numpy.ndarray): The alpha matte.
        blur_radius (int, optional): The blur radius for the fusion. Default is 90.

    Returns:
        numpy.ndarray: The estimated foreground image.
    """
    alpha = alpha[:, :, None]
    foreground, blurred_background = fb_blur_fusion_foreground_estimator(
        image, image, image, alpha, blur_radius
    )
    return fb_blur_fusion_foreground_estimator(
        image, foreground, blurred_background, alpha, blur_radius=6
    )[0]


def fb_blur_fusion_foreground_estimator(image, foreground, background, alpha, blur_radius=90):
    """
    Perform blur fusion to estimate the foreground and background images.

    Args:
        image (numpy.ndarray): The input image.
        foreground (numpy.ndarray): The initial foreground estimate.
        background (numpy.ndarray): The initial background estimate.
        alpha (numpy.ndarray): The alpha matte.
        blur_radius (int, optional): The blur radius for the fusion. Default is 90.

    Returns:
        tuple: A tuple containing the estimated foreground and blurred background images.
    """
    blurred_alpha = cv2.blur(alpha, (blur_radius, blur_radius))[:, :, None]

    blurred_foreground_alpha = cv2.blur(foreground * alpha, (blur_radius, blur_radius))
    blurred_foreground = blurred_foreground_alpha / (blurred_alpha + 1e-5)

    blurred_background_alpha = cv2.blur(background * (1 - alpha), (blur_radius, blur_radius))
    blurred_background = blurred_background_alpha / ((1 - blurred_alpha) + 1e-5)

    foreground = blurred_foreground + alpha * (
            image - alpha * blurred_foreground - (1 - alpha) * blurred_background
    )
    foreground = np.clip(foreground, 0, 1)

    return foreground, blurred_background


def infer_single(image_path, output_path, output_alpha_path, engine, context, transformers, numpy_to_pil):
    print(f"image path: {image_path}")
    origin_image = Image.open(image_path).convert("RGB")
    w, h = origin_image.size
    image_data = np.expand_dims(transformers(origin_image), axis=0).ravel()

    inputs, outputs, bindings, stream = common.allocate_buffers(engine)
    np.copyto(inputs[0].host, image_data)
    trt_outputs = common.do_inference(context, engine, bindings, inputs, outputs, stream)
    pred = np.squeeze(sigmoid(trt_outputs[-1].reshape((1, 1, 1024, 1024))))
    cropped_gray_image = numpy_to_pil(pred)
    predicted_alpha = Resize((h, w))(cropped_gray_image)

    predicted_alpha_array = np.array(predicted_alpha.convert('L')) / 255.0
    origin_image_array = np.array(origin_image) / 255.0
    estimated_foreground = fb_blur_fusion_foreground_estimator_2(origin_image_array, predicted_alpha_array)
    result = Image.fromarray((estimated_foreground * 255.0).astype(np.uint8))
    result.putalpha(predicted_alpha)
    result.save(output_path)
    predicted_alpha.save(output_alpha_path)
    common.free_buffers(inputs, outputs, stream)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-path", type=str, required=True, help="input path of image or input dir when mode=m")
    parser.add_argument("--output-path", type=str, required=True,
                        help="output path of result or output dir when mode=m")
    parser.add_argument("--output-alpha-path", type=str, required=True,
                        help="output alpha path or alpha die when mode=m")
    parser.add_argument("--engine-path", type=str, required=True, help="path of tensorrt engine")
    parser.add_argument("--process-size", type=int, default=1024)
    parser.add_argument("--mode", choices=["s", "m"], default="s")
    args = parser.parse_args()

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(load_engine(args.engine_path))
    context = engine.create_execution_context()

    transformers = Compose([
        Resize((1024, 1024)),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    numpy_to_pil = ToPILImage()

    if args.mode == "s":
        infer_single(args.image_path, args.output_path, args.output_alpha_path, engine, context, transformers,
                     numpy_to_pil)
    else:
        images = os.listdir(args.image_path)
        for image in images:
            print(f"image: {image}")
            image_path = os.path.join(args.image_path, image)
            output_image_path = os.path.join(args.output_path, image).replace(".jpg", ".png")
            output_alpha_path = os.path.join(args.output_alpha_path, image).replace(".jpg", ".png")
            infer_single(image_path, output_image_path, output_alpha_path, engine, context, transformers,
                         numpy_to_pil)


if __name__ == "__main__":
    main()
