import os
import torch
import numpy as np
from onnxruntime import InferenceSession
from PIL import Image, ImageDraw

def distance2bbox(points, deltas):
    x_ctr, y_ctr = points[:, 0], points[:, 1]
    l, t, r, b = deltas[:, 0], deltas[:, 1], deltas[:, 2], deltas[:, 3]
    x1, y1 = x_ctr - l, y_ctr - t
    x2, y2 = x_ctr + r, y_ctr + b
    return np.stack([x1, y1, x2, y2], axis=1)

def resize_pad(img: Image.Image, size: int, pad_value=(0, 0, 0)):
    w, h = img.size
    scale = size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = img.resize((new_w, new_h), Image.BILINEAR)
    pad_w = (size - new_w) // 2
    pad_h = (size - new_h) // 2
    canvas = Image.new(img.mode, (size, size), pad_value)
    canvas.paste(img_resized, (pad_w, pad_h))
    return canvas, (pad_h, size - new_h - pad_h, pad_w, size - new_w - pad_w)

class AdvancedAnimeSeg:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "refine_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "enable_refine": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MASK", "MASK")
    RETURN_NAMES = ("Refined Mask", "Coarse Mask")
    FUNCTION = "segment"
    CATEGORY = "Anime Segmentation"
    TITLE = "Advanced Anime Segmentation (ONNX)"

    def __init__(self):
        base = os.path.dirname(__file__)
        self.seg_model_path = os.path.join(base, "models", "anime_segmentor_rtmdet_e60_simplified.onnx")
        self.refine_model_path = os.path.join(base, "models", "mask_refiner_isnetdis_refine_last_simplified.onnx")
        self.session = InferenceSession(self.seg_model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

        self.refine_session = InferenceSession(self.refine_model_path, providers=["CPUExecutionProvider"])
        self.refine_input = self.refine_session.get_inputs()[0].name
        self.refine_outputs = [o.name for o in self.refine_session.get_outputs()]

    def segment(self, image, threshold=0.3, refine_threshold=0.3, enable_refine=True):
        src_arr = (image[0] * 255).to(torch.uint8).numpy()
        H_orig, W_orig = src_arr.shape[:2]
        src = Image.fromarray(src_arr, 'RGB')

        seg_H, seg_W = self.session.get_inputs()[0].shape[2:4]
        src.thumbnail((seg_W, seg_H), Image.Resampling.LANCZOS)
        pad_w, pad_h = (seg_W - src.width)//2, (seg_H - src.height)//2
        canvas = Image.new('RGB', (seg_W, seg_H), (114,114,114))
        canvas.paste(src, (pad_w, pad_h))

        arr = np.array(canvas)[:, :, ::-1].astype(np.float32)
        mean = np.array([123.675, 116.28, 103.53], np.float32)
        std  = np.array([58.395, 57.12, 57.375], np.float32)
        inp = ((arr - mean) / std).transpose(2,0,1)[None]

        out = dict(zip(self.output_names, self.session.run(self.output_names, {self.input_name: inp})))

        best_score = -np.inf
        best = None
        strides = [8, 16, 32]
        for i, s in enumerate(strides):
            logits = out[f'scores.stride{s}'][0,0]
            probs  = 1/(1+np.exp(-logits))
            idx = np.unravel_index(np.argmax(probs), probs.shape)
            if probs[idx] > best_score:
                best_score = probs[idx]
                best = (i, idx)
        if best is None:
            zero_mask = torch.zeros((1, H_orig, W_orig), dtype=torch.float32)
            return (zero_mask, zero_mask)

        si, (ry, cx) = best
        stride = strides[si]
        x_anchor = (cx) * stride
        y_anchor = (ry) * stride

        deltas = out[f'bboxes.stride{stride}'][0, :, ry, cx]
        box = distance2bbox(np.array([[x_anchor, y_anchor]]), deltas[np.newaxis,:])[0]
        x1, y1, x2, y2 = box
        x1 = max((x1 - pad_w) * W_orig/ (seg_W-2*pad_w), 0)
        x2 = min((x2 - pad_w) * W_orig/ (seg_W-2*pad_w), W_orig)
        y1 = max((y1 - pad_h) * H_orig/ (seg_H-2*pad_h), 0)
        y2 = min((y2 - pad_h) * H_orig/ (seg_H-2*pad_h), H_orig)

        center_x, center_y = (x1 + x2) * 0.5, (y1 + y2) * 0.5

        proto  = out['mask_proto'][0]
        kernel = out[f'coeffs.stride{stride}'][0, :, ry, cx]
        proto_tensor = torch.from_numpy(proto).unsqueeze(0)
        Hp, Wp = proto_tensor.shape[-2:]
        yv, xv = torch.meshgrid(
            torch.arange(Hp, dtype=torch.float32),
            torch.arange(Wp, dtype=torch.float32), indexing='ij')
        center_x_proto = center_x / stride
        center_y_proto = center_y / stride
        rel_x = (center_x_proto - xv) / (stride * 8.0)
        rel_y = (center_y_proto - yv) / (stride * 8.0)
        rel_coords = torch.stack((rel_x, rel_y), dim=0).unsqueeze(0).to(proto_tensor.dtype)
        mask_feat = torch.cat([rel_coords, proto_tensor], dim=1)

        kernels_tensor = torch.from_numpy(kernel).unsqueeze(0)
        weights, biases = self.parse_dynamic_params(kernels_tensor, proto_tensor.shape[1])
        x = mask_feat
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = torch.nn.functional.conv2d(x, w, bias=b, stride=1, padding=0, groups=1)
            if i < len(weights) - 1:
                x = torch.relu(x)
        mask_logits = x[0,0].cpu().numpy()
        mask_prob   = 1/(1+np.exp(-mask_logits))
        mask_bin    = (mask_prob > threshold).astype(np.uint8) * 255

        mask_img    = Image.fromarray(mask_bin).resize((seg_W, seg_H), Image.BILINEAR)
        mask_crop   = mask_img.crop((pad_w, pad_h, pad_w + src.width, pad_h + src.height))
        mask_final  = mask_crop.resize((W_orig, H_orig), Image.BILINEAR)

        coarse_tensor = torch.from_numpy(np.array(mask_final)).unsqueeze(0).float() / 255.0

        if enable_refine:
            refined = self._refine_mask(src_arr, mask_final, H_orig, W_orig, refine_threshold)
        else:
            refined = coarse_tensor

        return (refined, coarse_tensor)

    def _refine_mask(self, src_arr, mask_pil, H_orig, W_orig, threshold):
        refine_size = self.refine_session.get_inputs()[0].shape[2]
        img_pad, (pt, pb, pl, pr) = resize_pad(Image.fromarray(src_arr), refine_size, pad_value=(0,0,0))
        img_np = np.array(img_pad).astype(np.float32) / 255.0
        seg_pad, _ = resize_pad(mask_pil.convert('L'), refine_size, pad_value=0)
        seg_np = np.array(seg_pad).astype(np.float32) / 255.0
        inp = np.concatenate([img_np.transpose(2,0,1), seg_np[None]], axis=0)[None]
        out = self.refine_session.run(self.refine_outputs, {self.refine_input: inp})
        logits = out[0][0,0,:,:]
        print("[Refine] Logits range:", logits.min(), logits.max())
        prob = 1 / (1 + np.exp(-np.clip(logits, -50, 50)))
        prob_crop = prob[pt:refine_size-pb, pl:refine_size-pr]
        mask_arr = (prob_crop > threshold).astype(np.uint8) * 255
        mask = Image.fromarray(mask_arr.astype(np.uint8)).resize((W_orig, H_orig), Image.BILINEAR)
        print("[Refine] Final mask size:", mask.size)
        return torch.from_numpy(np.array(mask)).unsqueeze(0).float() / 255.0

    def parse_dynamic_params(self, flatten_kernels: torch.Tensor, proto_channels: int):
        n_inst = flatten_kernels.size(0)
        in_ch  = proto_channels + 2
        inter  = 8
        w_nums = [inter * in_ch, inter * inter, 1 * inter]
        b_nums = [inter, inter, 1]
        splits = torch.split(flatten_kernels, w_nums + b_nums, dim=1)
        w_splits, b_splits = splits[:3], splits[3:]
        weights, biases = [], []
        for i, (w, b) in enumerate(zip(w_splits, b_splits)):
            out_ch = b_nums[i]
            if i < 2:
                in_ch_layer = w_nums[i] // out_ch
                weights.append(w.view(n_inst * out_ch, in_ch_layer, 1, 1))
            else:
                weights.append(w.view(n_inst * out_ch, inter, 1, 1))
            biases.append(b.view(n_inst * out_ch))
        return weights, biases

NODE_CLASS_MAPPINGS = {"AdvancedAnimeSeg": AdvancedAnimeSeg}
NODE_DISPLAY_NAME_MAPPINGS = {"AdvancedAnimeSeg": AdvancedAnimeSeg.TITLE}
