from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import transforms

# 必須從 BiRefNet GitHub repository 根目錄執行
from models.birefnet import BiRefNet
from utils import check_state_dict


# ============================================================
# Settings
# ============================================================

WEIGHT_PATH = Path(
    "./BiRefNet_dynamic-general-epoch_174.pth"
)

# Dynamic 模型的輸入寬高需要是 32 的倍數
SIZE_MULTIPLE = 32

# 避免 4K、8K 圖片佔用過多記憶體
# 設成 None 表示不限制最長邊
MAX_SIDE = 2304

# Mac MPS 是否使用 FP16
# 如果之後遇到 MPS operator 不支援 half，可改成 False
USE_FP16_ON_MPS = True


# ============================================================
# Device
# ============================================================

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)

torch.set_float32_matmul_precision("high")


# ============================================================
# Load local .pth weights
# ============================================================

if not WEIGHT_PATH.is_file():
    raise FileNotFoundError(
        "找不到模型權重：{}".format(
            WEIGHT_PATH.resolve()
        )
    )

# 不另外下載 backbone，完整權重已包含於 .pth
birefnet = BiRefNet(bb_pretrained=False)

try:
    checkpoint = torch.load(
        WEIGHT_PATH,
        map_location="cpu",
        weights_only=True,
    )
except TypeError:
    # 相容較舊的 PyTorch
    checkpoint = torch.load(
        WEIGHT_PATH,
        map_location="cpu",
    )

# 某些 checkpoint 會將權重放在 state_dict 或 model 中
if isinstance(checkpoint, dict):
    if (
        "state_dict" in checkpoint
        and isinstance(checkpoint["state_dict"], dict)
    ):
        checkpoint = checkpoint["state_dict"]

    elif (
        "model" in checkpoint
        and isinstance(checkpoint["model"], dict)
    ):
        checkpoint = checkpoint["model"]

state_dict = check_state_dict(checkpoint)

# 確認模型架構和權重完全相符
birefnet.load_state_dict(
    state_dict,
    strict=True,
)

birefnet.eval()
birefnet.to(device)


# ============================================================
# Precision
# ============================================================

if device.type == "cuda":
    use_fp16 = True
elif device.type == "mps":
    use_fp16 = USE_FP16_ON_MPS
else:
    use_fp16 = False

if use_fp16:
    birefnet.half()
else:
    birefnet.float()

print(
    "Loaded weights:",
    WEIGHT_PATH.resolve(),
)

print(
    "Inference dtype:",
    "float16" if use_fp16 else "float32",
)


# ============================================================
# Image preprocessing
# ============================================================

normalize_image = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def floor_to_multiple(
    value: int,
    multiple: int = 32,
) -> int:
    """
    將數值向下調整成 multiple 的倍數。

    例如：
        1365 -> 1344
        2048 -> 2048
    """
    return max(
        multiple,
        value // multiple * multiple,
    )


def calculate_inference_size(
    image_size: Tuple[int, int],
    max_side: Optional[int] = 2304,
    multiple: int = 32,
) -> Tuple[int, int]:
    """
    計算 dynamic 模型使用的推論尺寸。

    1. 保持原始長寬比例縮放。
    2. 最長邊不超過 max_side。
    3. 寬和高都向下對齊到 32 的倍數。

    Pillow 的尺寸格式為：
        (width, height)
    """
    original_width, original_height = image_size

    resized_width = original_width
    resized_height = original_height

    if max_side is not None:
        longest_side = max(
            resized_width,
            resized_height,
        )

        if longest_side > max_side:
            scale = max_side / float(longest_side)

            resized_width = max(
                1,
                round(resized_width * scale),
            )
            resized_height = max(
                1,
                round(resized_height * scale),
            )

    aligned_width = floor_to_multiple(
        resized_width,
        multiple,
    )
    aligned_height = floor_to_multiple(
        resized_height,
        multiple,
    )

    return aligned_width, aligned_height


def prepare_inference_image(
    image: Image.Image,
    max_side: Optional[int] = 2304,
    multiple: int = 32,
) -> Image.Image:
    """
    將圖片調整成 dynamic BiRefNet 可接受的尺寸。
    """
    inference_size = calculate_inference_size(
        image_size=image.size,
        max_side=max_side,
        multiple=multiple,
    )

    if inference_size == image.size:
        return image

    return image.resize(
        inference_size,
        resample=Image.Resampling.LANCZOS,
    )


# ============================================================
# Inference
# ============================================================

def extract_object(
    model,
    imagepath,
    mask_output_path=None,
    rgba_output_path=None,
    max_side=2304,
):
    imagepath = Path(imagepath)

    if not imagepath.is_file():
        raise FileNotFoundError(
            "找不到輸入圖片：{}".format(
                imagepath.resolve()
            )
        )

    # 修正手機照片 EXIF 旋轉資訊
    with Image.open(imagepath) as loaded_image:
        original_image = ImageOps.exif_transpose(
            loaded_image
        ).convert("RGB")

    original_width, original_height = (
        original_image.size
    )

    # 縮放並將寬高調整成 32 的倍數
    inference_image = prepare_inference_image(
        original_image,
        max_side=max_side,
        multiple=SIZE_MULTIPLE,
    )

    inference_width, inference_height = (
        inference_image.size
    )

    print(
        "Original size:",
        original_image.size,
    )
    print(
        "Inference size:",
        inference_image.size,
    )

    if (
        inference_width % SIZE_MULTIPLE != 0
        or inference_height % SIZE_MULTIPLE != 0
    ):
        raise ValueError(
            "推論尺寸必須為 {} 的倍數，"
            "目前尺寸為：{}".format(
                SIZE_MULTIPLE,
                inference_image.size,
            )
        )

    input_tensor = normalize_image(
        inference_image
    ).unsqueeze(0)

    input_tensor = input_tensor.to(device)

    if use_fp16:
        input_tensor = input_tensor.half()
    else:
        input_tensor = input_tensor.float()

    with torch.inference_mode():
        model_outputs = model(input_tensor)

        # BiRefNet 最後一個輸出是主要 segmentation logits
        logits = model_outputs[-1]

        # sigmoid 後是 0～1 的 soft mask
        pred = logits.sigmoid().float()

        # 直接在 tensor 浮點狀態下恢復成原圖尺寸
        pred = F.interpolate(
            pred,
            size=(
                original_height,
                original_width,
            ),
            mode="bilinear",
            align_corners=False,
        )

        pred = (
            pred[0, 0]
            .clamp(0.0, 1.0)
            .cpu()
        )

    # float 0～1 -> uint8 0～255
    mask_array = (
        pred.numpy() * 255.0
    ).round().astype(np.uint8)

    # L mode：8-bit grayscale
    mask = Image.fromarray(
        mask_array,
        mode="L",
    )

    # 儲存 grayscale alpha mask
    if mask_output_path is not None:
        mask_path = Path(mask_output_path)

        mask_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        mask.save(
            mask_path,
            format="PNG",
        )

        print(
            "Saved mask:",
            mask_path.resolve(),
        )

    # 建立 RGBA 透明 PNG
    rgba_image = original_image.convert("RGBA")
    rgba_image.putalpha(mask)

    if rgba_output_path is not None:
        rgba_path = Path(rgba_output_path)

        rgba_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        rgba_image.save(
            rgba_path,
            format="PNG",
        )

        print(
            "Saved RGBA:",
            rgba_path.resolve(),
        )

    return rgba_image, mask


# ============================================================
# Run
# ============================================================

rgba_image, mask = extract_object(
    birefnet,
    imagepath="IMG_3405.JPG",
    mask_output_path=(
        "outputs/777_IMG_3405_mask.png"
    ),
    rgba_output_path=(
        "outputs/777_IMG_3405_transparent.png"
    ),
    max_side=MAX_SIDE,
)

print("Done.")