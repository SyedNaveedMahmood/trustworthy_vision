"""Geometry and explicitly unvalidated occlusion/blur operators."""
import math
import numpy as np
from PIL import Image, ImageFilter, ImageDraw


def xywh_to_xyxy(box, width, height):
    x, y, w, h = map(float, box)
    if not all(math.isfinite(v) for v in (x,y,w,h)) or w <= 0 or h <= 0:
        raise ValueError("Invalid box")
    result = (max(0, x), max(0, y), min(width, x+w), min(height, y+h))
    if result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError("Empty clipped box")
    return result


def resize_box(box, original_size, new_size):
    sx, sy = new_size[0]/original_size[0], new_size[1]/original_size[1]
    return tuple(v*s for v,s in zip(box, (sx,sy,sx,sy)))


def box_mask(size, box):
    x0,y0,x1,y1 = box
    if not (0 <= x0 < x1 <= size[0] and 0 <= y0 < y1 <= size[1]):
        raise ValueError("Box outside image")
    a = np.zeros((size[1],size[0]), dtype=np.uint8)
    a[math.floor(y0):math.ceil(y1),math.floor(x0):math.ceil(x1)] = 255
    return Image.fromarray(a)


def edit(image, box, operator, sham=False, sigma=24):
    image = image.convert("RGB")
    mask = box_mask(image.size, box)
    if operator == "occlusion":
        mean = tuple(np.asarray(image).reshape(-1,3).mean(axis=0).round().astype(int))
        modified = Image.new("RGB", image.size, mean)
        if sham:
            # Contour is inside the box. Interior preserved, boundary not semantically guaranteed.
            inner = mask.filter(ImageFilter.MinFilter(5))
            mask = Image.fromarray(np.asarray(mask)-np.asarray(inner))
    elif operator == "blur":
        modified = image.filter(ImageFilter.GaussianBlur(1 if sham else sigma))
    elif operator == "lama":
        raise RuntimeError("Official LaMa environment/checkpoint not validated: no fallback")
    else:
        raise ValueError("Unknown operator")
    # Strictly localized composite. Feathering requires a dev amendment and audit.
    return Image.composite(modified, image, mask)


def overlay_boxes(image, boxes):
    if len(boxes) != 4:
        raise ValueError("Four candidate boxes required")
    image = image.copy()
    draw = ImageDraw.Draw(image)
    for label, box in zip("ABCD", boxes):
        draw.rectangle(box, outline="white", width=2)
        draw.text((box[0]+3,box[1]+3), label, fill="black", stroke_width=2, stroke_fill="white")
    return image
