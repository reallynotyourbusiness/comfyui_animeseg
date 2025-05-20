# Simple custom node to segment anime images using https://github.com/SkyTNT/anime-segmentation
# To install the custom node, copy this folder into your `ComfyUI/custom_nodes` directory
# Place the `isnetis.onnx` file inside the `models` subfolder
# Requires: pip install onnxruntime

import torch
import numpy as np
from PIL import Image
from onnxruntime import InferenceSession
import os

class AnimeSeg:
    def __init__(self):
        # Load model only once on init from the models folder
        model_path = os.path.join(os.path.dirname(__file__), "models", "isnetis.onnx")
        self.session = InferenceSession(model_path, providers=["CPUExecutionProvider"])

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "segment"
    CATEGORY = "Anime Segmentation"
    TITLE = "Simple Anime Segmentation"

    def get_mask(self, src):
        raw_size = src.size
        seg_size = self.session.get_inputs()[0].shape[2:4]
        src.thumbnail(seg_size)
        dst = ((seg_size[0] - src.size[0]) // 2, (seg_size[1] - src.size[1]) // 2)

        img = Image.new('RGB', seg_size, (0,0,0))
        img.paste(src, dst)
        arr = np.array(img)
        arr = arr[:, :, ::-1].transpose((2,0,1))[None].astype(np.float32) / 255.0

        in_name = self.session.get_inputs()[0].name
        out_name = self.session.get_outputs()[0].name
        mask = self.session.run([out_name], {in_name: arr})[0]

        mask_tensor = torch.from_numpy(mask).clamp(0.0,1.0).transpose(3,2)
        mask_tensor = mask_tensor[:,:,dst[0]:seg_size[0]-dst[0], dst[1]:seg_size[1]-dst[1]]
        mask_tensor = torch.nn.functional.interpolate(mask_tensor, raw_size, mode="bilinear")
        return mask_tensor.transpose(3,2)

    def segment(self, image):
        img = Image.fromarray((image[0]*255).to(torch.uint8).numpy(), 'RGB')
        mask = self.get_mask(img)
        return (mask,)

NODE_CLASS_MAPPINGS = {"SimpleAnimeSeg": AnimeSeg}
NODE_DISPLAY_NAME_MAPPINGS = {"SimpleAnimeSeg": AnimeSeg.TITLE}
