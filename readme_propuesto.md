# ComfyUI Anime Segmentation Nodes v1.2.0

This is a set of custom nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) that provide anime-style image segmentation using efficient pre-trained models. It includes:

- ✅ `SimpleAnimeSeg` — fast ONNX-based segmentation using SkyTNT's model.
- ✅ `AdvancedAnimeSeg` — instance-based character segmentation with converted versions of the CartoonSegmentation models provided by dreMaz.

As of now, this entire node pack is CPU-oriented. GPU support may be added in later versions.

> ⚠️ These nodes are not perfect. Anime character segmentation is still a tricky problem. While decent, results may vary depending on character appearance and background complexity.

---

## Features

### ✅ SimpleAnimeSeg

This provides quick segmentation and is often sufficient when there's a clear difference between the character and background. However, it can struggle when the character's clothes or skin match background elements closely. In such cases, using the advanced node is recommended.

- Fast CPU-friendly segmentation
- Uses ONNX model `isnetis.onnx`
- Lightweight and easy to integrate
- **Supports batch processing**

![Simple segmentation example](images/Simple%20segmentation%20example.png)
![Simple segmentation problems](images/Simple%20Segmentation%20problems.png)

---

### ✅ AdvancedAnimeSeg

A more robust segmentation node based on the [CartoonSegmentation](https://github.com/CartoonSegmentation/CartoonSegmentation) project. This version provides options such as threshold adjustment to refine results.

It uses the original model weights from [dreMaz/AnimeInstanceSegmentation](https://huggingface.co/dreMaz/AnimeInstanceSegmentation), but relies on ONNX-converted versions found [here](https://huggingface.co/Faor-Mati/anime-character-segmentation/tree/main).

> While more customizable, this node can still fail in specific edge cases where the simpler model performs better. It's called "advanced" due to its flexibility and multi-stage approach — not guaranteed accuracy.

- Combines RTMDet-based detection with ISNet-based mask refinement
- Pure PyTorch (no mmcv/mmdet dependencies)
- Cleaner masks, great for downstream editing (e.g., outfits)
- Threshold option for finer control
- **Supports batch processing**

![Advanced segmentation example](images/advanced%20segmentation.png)

---

## Installation

1.  **Clone the repository:**
    Navigate to your `ComfyUI/custom_nodes` directory and run:
    ```bash
    git clone https://github.com/your-repo-url.git comfyui_animeseg
    ```
    (Replace `https://github.com/your-repo-url.git` with the actual repository URL)

2.  **Download the models:**
    Download the following model files and place them in the `comfyui_animeseg/models` directory (create the `models` folder if it doesn't exist):
    - [`isnetis.onnx`](https://huggingface.co/skytnt/anime-seg/resolve/main/isnetis.onnx) (for `SimpleAnimeSeg`)
    - [`ranime_segmentor_rtmdet_e60_simplified.onnx`](https://huggingface.co/Faor-Mati/anime-character-segmentation/resolve/main/anime_segmentor_rtmdet_e60_simplified.onnx) (for `AdvancedAnimeSeg` detector)
    - [`mask_refiner_isnetdis_refine_last_simplified.onnx`](https://huggingface.co/Faor-Mati/anime-character-segmentation/resolve/main/mask_refiner_isnetdis_refine_last_simplified.onnx) (for `AdvancedAnimeSeg` refiner)

3.  **Install dependencies:**
    Make sure you have the required packages installed in your ComfyUI's Python environment. You can install them using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

Both `SimpleAnimeSeg` and `AdvancedAnimeSeg` nodes take an image batch as input and output a mask batch. You can use them in your workflows to separate characters from the background.

1.  **Add the node:** Right-click on the canvas in ComfyUI, go to "Add Node" -> "Anime Segmentation", and select either `SimpleAnimeSeg` or `AdvancedAnimeSeg`.
2.  **Connect the image:** Connect an image or a batch of images to the `image` input of the node.
3.  **Get the mask:** The node will output a `mask` that you can use for other operations like inpainting, background replacement, etc.

---

## Changelog

### v1.2.0
- ✨ **Feature:** Both `SimpleAnimeSeg` and `AdvancedAnimeSeg` now support batch processing. You can feed them a batch of images and they will return a corresponding batch of masks.

### v1.1.0
- Initial release.

---

## Future Plans

- 🚧 Merge both nodes to generate higher quality composite masks
- 🧠 Train a part-based segmenter for individual character parts (legs, arms, hair, clothing, etc)
- ♻️ Refactor node structure for more flexibility (may break backward compatibility)

These plans are exploratory. No timeline is guaranteed.

---

## Credits

- [SkyTNT](https://github.com/SkyTNT) — ONNX segmentation model
- [city96](https://github.com/city96) — original segmentation node gist
- [dreMaz](https://github.com/dmMaze) and [ljsabc](https://github.com/ljsabc) — advanced node model providers
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) — node framework

---

Happy generations! 🎨✨
