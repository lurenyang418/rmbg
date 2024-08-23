# coding:utf-8

import numpy as np
import onnxruntime as ort
from PIL import Image


def normalize(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32, copy=False)
    img /= 255.0
    img -= np.array([0.5, 0.5, 0.5])
    img /= np.array([1.0, 1.0, 1.0])
    return img


def preprocess(im: Image.Image) -> np.ndarray:
    im_1024 = im.resize((1024, 1024))
    img = np.array(im_1024)
    img = normalize(img)
    img = np.expand_dims(img.transpose(2, 0, 1), axis=0).astype("float32")

    return img


def postprocess(mask: np.ndarray) -> np.ndarray:
    mask = mask[0][0]
    ma = np.max(mask)
    mi = np.min(mask)
    mask = (mask - mi) / (ma - mi)
    mask = (mask * 255).astype(np.uint8)

    return mask


def predict(img_file: str) -> Image.Image:
    im = Image.open(img_file)
    # 调整至模型推理所需的 1024 * 1024
    img = preprocess(im)
    sess = ort.InferenceSession("models/model_fp16.onnx")
    session_inputs = sess.get_inputs()
    input_dict = {session_inputs[0].name: img}

    mask = sess.run(None, input_dict)[0]
    mask = postprocess(mask)
    mask_im = Image.fromarray(mask).resize(im.size)
    # 添加透明度通道
    img = Image.new("RGBA", mask_im.size, (0, 0, 0, 0))
    img.paste(im, mask=mask_im)

    return img


if __name__ == "__main__":
    im = predict("example_input.jpg")
    im.save("example_output.png")
