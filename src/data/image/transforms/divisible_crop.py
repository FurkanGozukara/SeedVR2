# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.

import os
from typing import Union
import torch
from PIL import Image
from torchvision.transforms import functional as TVF


class DivisibleCrop:
    def __init__(self, factor):
        if not isinstance(factor, tuple):
            factor = (factor, factor)

        self.height_factor, self.width_factor = factor[0], factor[1]

    def __call__(self, image: Union[torch.Tensor, Image.Image]):
        if isinstance(image, torch.Tensor):
            height, width = image.shape[-2:]
        elif isinstance(image, Image.Image):
            width, height = image.size
        else:
            raise NotImplementedError

        cropped_height = height - (height % self.height_factor)
        cropped_width = width - (width % self.width_factor)

        image = TVF.center_crop(img=image, output_size=(cropped_height, cropped_width))
        return image


class DivisiblePad:
    """
    Pad image to make dimensions divisible by a factor.
    Pads with black (0) to avoid data loss.
    """
    def __init__(self, factor):
        if not isinstance(factor, tuple):
            factor = (factor, factor)
        self.height_factor, self.width_factor = factor[0], factor[1]

    def __call__(self, image: Union[torch.Tensor, Image.Image]):
        if isinstance(image, torch.Tensor):
            height, width = image.shape[-2:]
        elif isinstance(image, Image.Image):
            width, height = image.size
        else:
            raise NotImplementedError

        # Calculate padding needed
        pad_height = (self.height_factor - (height % self.height_factor)) % self.height_factor
        pad_width = (self.width_factor - (width % self.width_factor)) % self.width_factor

        if pad_height == 0 and pad_width == 0:
            return image

        # Pad bottom/right. SEEDVR2_PAD_MODE=reflect mirrors the edge rows/columns instead
        # of black (Flirt-Inc f69b8e749a / ussoewwin 874e6e36d9); the padding is cropped
        # from the output either way.
        if isinstance(image, torch.Tensor):
            # Pad format: (left, right, top, bottom)
            padding = (0, pad_width, 0, pad_height)
            pad_mode = os.environ.get("SEEDVR2_PAD_MODE", "constant").strip().lower()
            if pad_mode in ("reflect", "replicate") and image.ndim >= 3:
                if pad_mode == "reflect" and (pad_width >= width or pad_height >= height):
                    pad_mode = "replicate"
                image = torch.nn.functional.pad(image, padding, mode=pad_mode)
            else:
                image = torch.nn.functional.pad(image, padding, mode='constant', value=0.0)
        elif isinstance(image, Image.Image):
            new_width = width + pad_width
            new_height = height + pad_height
            result = Image.new(image.mode, (new_width, new_height), (0, 0, 0))
            result.paste(image, (0, 0))
            image = result

        return image
