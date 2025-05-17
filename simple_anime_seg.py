#
# Simple custom node to segment anime images using https://github.com/SkyTNT/anime-segmentation
# To install the custom node, copy this file to your `ComfyUI/custom_nodes` folder
# Place the `isnetis.onnx` file in the same folder or adjust the path below
# Requires: pip install onnxruntime
#

import torch
import numpy as np
from PIL import Image
from onnxruntime import InferenceSession
import os

class AnimeSeg:
    def __init__(self):
        # Load model only once on init
        model_path = os.path.join(os.path.dirname(__file__), "isnetis.onnx")
        self.session = InferenceSession(model_path, providers=["CPUExecutionProvider"])

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "segment"
    CATEGORY = "bootleg"
    TITLE = "Anime Segmentation"

    def get_mask(self, src):
        raw_size = src.size
        seg_size = self.session.get_inputs()[0].shape[2:4]
        src.thumbnail(seg_size)
        dst_size = int((seg_size[0] - src.size[0]) / 2), int((seg_size[1] - src.size[1]) / 2)

        img = Image.new('RGB', size=seg_size, color=(0, 0, 0))
        img.paste(src, dst_size)
        img = np.array(img)
        img = img[:, :, ::-1]  # PIL RGB to OpenCV BGR
        img = img.transpose((2, 0, 1))  # N, C, H, W
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, 0)

        in_name = self.session.get_inputs()[0].name
        out_name = self.session.get_outputs()[0].name
        mask = self.session.run([out_name], {in_name: img})[0]

        mask = torch.clamp(torch.from_numpy(mask), 0.0, 1.0).transpose(3, 2)
        mask = mask[:, :, dst_size[0]:(seg_size[0] - dst_size[0]), dst_size[1]:(seg_size[1] - dst_size[1])]
        mask = torch.nn.functional.interpolate(mask, raw_size, mode="bilinear")
        return mask.transpose(3, 2)

    def segment(self, image):
        img = Image.fromarray((image[0] * 255.0).to(torch.uint8).numpy(), mode='RGB')
        mask = self.get_mask(img)
        return (mask,)

NODE_CLASS_MAPPINGS = {"SimpleAnimeSeg": AnimeSeg}
NODE_DISPLAY_NAME_MAPPINGS = {"SimpleAnimeSeg": AnimeSeg.TITLE}
