import io
import os
import time
import math
import json
import base64
from datetime import datetime

from flask import Flask, request, jsonify, send_from_directory, abort

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn.functional as F

from transformers import (
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
    AutoImageProcessor,
    AutoModelForDepthEstimation,
)

from ultralytics import YOLO


# ----------------------------
#  Конфигурация
# ----------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEBUG_DIR = os.path.join(os.path.dirname(__file__), "debug")
os.makedirs(DEBUG_DIR, exist_ok=True)

# какие классы SegFormer считаем "проходимыми"
FREE_LABELS = {
    "floor",
    "ground",
    "road",
    "sidewalk",
    "path",
    "runway",
}

# YOLO: какие классы нам интересны (остальные можно игнорить)
YOLO_CLASSES_WHITELIST = {
    "person",
    "bicycle",
    "motorbike",
    "car",
    "bus",
    "truck",
    "bench",
    "chair",
    "couch",
    "sofa",
    "bed",
    "dining table",
    "tv",
    "tvmonitor",
    "stop sign",
    "traffic light",
    "parking meter",
    "potted plant",
    "refrigerator",
    "chair",
}

# Простая мапа английский → русский для озвучки
YOLO_LABELS_RU = {
    "person": "человек",
    "bicycle": "велосипед",
    "motorbike": "мотоцикл",
    "car": "машина",
    "bus": "автобус",
    "truck": "грузовик",
    "bench": "скамейка",
    "chair": "стул",
    "couch": "диван",
    "sofa": "диван",
    "bed": "кровать",
    "dining table": "стол",
    "tv": "телевизор",
    "tvmonitor": "телевизор",
    "stop sign": "знак",
    "traffic light": "светофор",
    "parking meter": "паркомат",
    "potted plant": "растение",
    "refrigerator": "холодильник",
}


app = Flask(__name__)


# ----------------------------
#  Загрузка моделей
# ----------------------------

print("[init] Using device:", DEVICE)

print("[init] Loading SegFormer...")
seg_processor = SegformerImageProcessor.from_pretrained(
    "nvidia/segformer-b0-finetuned-ade-512-512"
)
seg_model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b0-finetuned-ade-512-512"
).to(DEVICE)
seg_model.eval()
seg_id2label = seg_model.config.id2label

print("[init] Loading ZoeDepth...")
depth_processor = AutoImageProcessor.from_pretrained("Intel/zoedepth-nyu-kitti")
depth_model = AutoModelForDepthEstimation.from_pretrained("Intel/zoedepth-nyu-kitti").to(
    DEVICE
)
depth_model.eval()

print("[init] Loading YOLOv8x (COCO)...")
yolo_model = YOLO("yolov8x.pt")  # самая мощная из стандартных v8
yolo_model.to(DEVICE)


# ----------------------------
#  Вспомогательные функции
# ----------------------------

def current_ts_for_filename() -> str:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    return ts


def pil_to_numpy_rgb(img: Image.Image) -> np.ndarray:
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.array(img)


def run_zoedepth(image_pil: Image.Image) -> np.ndarray:
    """
    Запускаем ZoeDepth и возвращаем depth-карту (H, W) в float32.
    Без post_process_depth_estimation, чтобы не ловить ошибки.
    """
    t0 = time.time()

    inputs = depth_processor(images=image_pil, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = depth_model(**inputs)

    if hasattr(outputs, "predicted_depth"):
        depth = outputs.predicted_depth
    else:
        depth = outputs

    if depth.ndim == 4:
        depth = depth[0, 0]
    elif depth.ndim == 3:
        depth = depth[0]
    else:
        depth = depth.squeeze()

    orig_w, orig_h = image_pil.size
    depth = depth.unsqueeze(0).unsqueeze(0)  # (1,1,h,w)
    depth_resized = F.interpolate(
        depth,
        size=(orig_h, orig_w),
        mode="bilinear",
        align_corners=False,
    )[0, 0]

    depth_np = depth_resized.detach().cpu().numpy().astype("float32")

    t1 = time.time()
    app.logger.info("[depth] zoedepth forward+postprocess %.3f s", t1 - t0)

    return depth_np


def run_segformer(image_pil: Image.Image) -> tuple[np.ndarray, float, float, int]:
    """
    Запускаем SegFormer, возвращаем:
      seg_mask: (H, W) int32 – id класса для каждого пикселя
      free_frac: доля пикселей, входящих в "проходимые" классы
      obs_frac: доля "непроходимых" пикселей
      num_pixels: всего пикселей
    """
    t0 = time.time()

    inputs = seg_processor(images=image_pil, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = seg_model(**inputs)

    t1 = time.time()
    app.logger.info("[segformer] forward %.3f s", t1 - t0)

    logits = outputs.logits  # (1, num_labels, h, w)
    orig_w, orig_h = image_pil.size

    t2 = time.time()
    logits_upsampled = F.interpolate(
        logits,
        size=(orig_h, orig_w),
        mode="bilinear",
        align_corners=False,
    )
    seg = logits_upsampled.argmax(dim=1)[0].detach().cpu().numpy().astype("int32")
    t3 = time.time()

    app.logger.info("[segformer] upsample+argmax %.3f s", t3 - t2)

    h, w = seg.shape
    num_pixels = h * w
    free_count = 0
    obs_count = 0

    for class_id in np.unique(seg):
        label = seg_id2label.get(int(class_id), "")
        mask = (seg == class_id)
        cnt = int(mask.sum())
        if label in FREE_LABELS:
            free_count += cnt
        else:
            obs_count += cnt

    free_frac = free_count / max(1, num_pixels)
    obs_frac = obs_count / max(1, num_pixels)

    return seg, free_frac, obs_frac, num_pixels


def direction_from_bbox(bbox, img_w) -> str:
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    rel = cx / img_w
    if rel < 0.33:
        return "слева"
    elif rel > 0.66:
        return "справа"
    else:
        return "по центру"


def estimate_distance_from_depth(depth: np.ndarray, bbox) -> float | None:
    """
    depth: (H, W) float32
    bbox: [x1, y1, x2, y2]
    Возвращаем медиану глубины внутри бокса.
    """
    h, w = depth.shape
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))
    if x2 <= x1 + 1 or y2 <= y1 + 1:
        return None

    patch = depth[y1:y2, x1:x2]
    patch = patch[np.isfinite(patch)]
    patch = patch[patch > 0]

    if patch.size == 0:
        return None

    med = float(np.median(patch))
    return med


def run_yolo(image_pil: Image.Image, depth: np.ndarray):
    """
    Запускаем YOLOv8x, возвращаем список объектов в нашем формате.
    """
    t0 = time.time()

    img_np = pil_to_numpy_rgb(image_pil)
    results = yolo_model(img_np, verbose=False)[0]

    t1 = time.time()
    boxes = results.boxes
    num_boxes = boxes.xyxy.shape[0] if boxes is not None else 0
    app.logger.info("[yolo] forward %.3f s, boxes=%d", t1 - t0, num_boxes)

    h, w = depth.shape
    objects = []

    if boxes is None or num_boxes == 0:
        app.logger.info("[yolo] filtered objects: 0")
        return objects, t1 - t0

    for b in boxes:
        conf = float(b.conf.item())
        cls_id = int(b.cls.item())
        label_en = results.names.get(cls_id, f"class_{cls_id}")

        if label_en not in YOLO_CLASSES_WHITELIST:
            continue
        if conf < 0.35:
            continue

        x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
        bbox = [x1, y1, x2, y2]
        area = max(1.0, (x2 - x1) * (y2 - y1))
        frac = area / float(w * h)

        dist = estimate_distance_from_depth(depth, bbox)
        direction = direction_from_bbox(bbox, w)

        label_ru = YOLO_LABELS_RU.get(label_en, label_en)

        objects.append(
            {
                "label": label_ru,
                "label_en": label_en,
                "direction": direction,
                "distance_m": dist,
                "fraction": frac,
                "bbox": [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))],
                "confidence": conf,
                "source": "yolo",
            }
        )

    app.logger.info("[yolo] filtered objects: %d", len(objects))
    return objects, t1 - t0


def build_speech_text(has_obstacles, objects, mode: str) -> str:
    """
    Формируем короткую фразу для озвучки.
    Формат как раньше: "Впереди стол, в полутора метрах".
    Без "чуть левее/правее".
    """

    if not has_obstacles or not objects:
        if mode == "outdoor":
            return "Впереди свободно"
        else:
            return "Впереди нет препятствий"

    nearest = None
    for obj in objects:
        d = obj.get("distance_m")
        if d is None or d <= 0:
            continue
        if nearest is None or d < nearest.get("distance_m", 9999):
            nearest = obj

    if nearest is None:
        return "Впереди препятствие"

    label = nearest.get("label", "объект")
    direction = nearest.get("direction")
    dist = nearest.get("distance_m", 0.0)

    if direction and direction not in ("по центру", "центр"):
        dir_part = f" {direction}"
    else:
        dir_part = ""

    if dist <= 0.6:
        phrase = f"Впереди{dir_part} {label} вплотную"
        return phrase
    elif dist < 1.0:
        phrase = f"Впереди{dir_part} {label}, менее метра"
        return phrase

    if 0.95 <= dist <= 1.05:
        dist_str = "одном метре"
    elif 1.45 <= dist <= 1.55:
        dist_str = "полутора метрах"
    elif 1.95 <= dist <= 2.05:
        dist_str = "двух метрах"
    else:
        dist_rounded = round(dist, 1)
        if 1.0 < dist_rounded <= 4.0:
            dist_str = f"{dist_rounded} метра"
        else:
            dist_str = f"{dist_rounded} метров"

    phrase = f"Впереди{dir_part} {label}, примерно на {dist_str}"
    return phrase


def save_debug_image(
    img: Image.Image,
    depth: np.ndarray | None,
    objects: list[dict],
    free_frac: float,
    obs_frac: float,
    mode: str,
) -> str:
    """
    Сохраняем debug-картинку с боксами и текстом, возвращаем имя файла.
    """

    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)

    # Боксы YOLO
    for obj in objects:
        bbox = obj.get("bbox")
        if not bbox:
            continue
        x1, y1, x2, y2 = bbox
        label = obj.get("label", "obj")
        dist = obj.get("distance_m")
        direction = obj.get("direction", "")
        txt_parts = [label]
        if direction:
            txt_parts.append(direction)
        if dist is not None:
            txt_parts.append(f"{dist:.1f}м")
        text = " / ".join(txt_parts)

        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 20)
        except Exception:
            font = ImageFont.load_default()

        bbox_text = draw.textbbox((0, 0), text, font=font)
        tw = bbox_text[2] - bbox_text[0]
        th = bbox_text[3] - bbox_text[1]

        tx = x1
        ty = max(0, y1 - th - 4)
        draw.rectangle([tx, ty, tx + tw + 4, ty + th + 4], fill=(0, 0, 0))
        draw.text((tx + 2, ty + 2), text, font=font, fill=(255, 255, 255))

    info_lines = [
        f"mode: {mode}",
        f"free: {free_frac:.3f}",
        f"obs: {obs_frac:.3f}",
        f"objects: {len(objects)}",
    ]
    info_text = "\n".join(info_lines)
    try:
        font_info = ImageFont.truetype("DejaVuSans.ttf", 18)
    except Exception:
        font_info = ImageFont.load_default()

    info_bbox = draw.multiline_textbbox((0, 0), info_text, font=font_info)
    iw = info_bbox[2] - info_bbox[0]
    ih = info_bbox[3] - info_bbox[1]

    draw.rectangle([0, 0, iw + 10, ih + 10], fill=(0, 0, 0, 160))
    draw.multiline_text((5, 5), info_text, font=font_info, fill=(255, 255, 255))

    filename = f"debug_{current_ts_for_filename()}.jpg"
    path = os.path.join(DEBUG_DIR, filename)
    img.save(path, "JPEG", quality=90)

    return filename


def get_last_debug_image() -> str | None:
    files = [f for f in os.listdir(DEBUG_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not files:
        return None
    files.sort()
    return files[-1]


# ----------------------------
#  Маршруты
# ----------------------------

@app.route("/analyze", methods=["POST"])
def analyze():
    t_start = time.time()
    request_id = str(int(t_start * 1000))

    mode = request.args.get("mode", "indoor")
    debug_flag = request.args.get("debug", "0") in ("1", "true", "True")

    t0 = time.time()
    raw = request.get_data()  # что реально пришло в body
    t1 = time.time()

    image = None
    image_bytes_for_log = raw

    # 1) Попытка: сырые байты — сразу картинка (как раньше)
    if raw:
        try:
            image = Image.open(io.BytesIO(raw))
        except Exception:
            image = None

    # 2) Если нет — пробуем multipart (request.files)
    if image is None and request.files:
        try:
            f = next(iter(request.files.values()))
            fb = f.read()
            image = Image.open(io.BytesIO(fb))
            image_bytes_for_log = fb
        except Exception:
            image = None

    # 3) Если всё ещё нет — пробуем JSON с base64
    if image is None and raw:
        try:
            text = raw.decode("utf-8", errors="ignore")
            obj = json.loads(text)
            img_b64 = obj.get("image") or obj.get("image_base64")
            if img_b64:
                fb = base64.b64decode(img_b64)
                image = Image.open(io.BytesIO(fb))
                image_bytes_for_log = fb
        except Exception:
            image = None

    if image is None:
        app.logger.error(
            "[analyze %s] failed to open image", request_id, exc_info=True
        )
        return jsonify({"error": "invalid image"}), 400

    image = image.convert("RGB")
    w, h = image.size

    app.logger.info(
        "[analyze %s] image loaded: %dx%d, bytes=%d, network_read=%.3fs, since_request=%.3fs",
        request_id,
        w,
        h,
        len(image_bytes_for_log) if image_bytes_for_log is not None else 0,
        t1 - t0,
        t1 - t_start,
    )

    # --- Depth ---
    depth = run_zoedepth(image)

    # --- SegFormer ---
    seg, free_frac, obs_frac, num_pixels = run_segformer(image)
    has_obstacles = obs_frac > 0.02

    # --- YOLO ---
    yolo_objects, yolo_time = run_yolo(image, depth)
    all_objects = yolo_objects

    # --- Debug ---
    debug_filename = None
    if debug_flag:
        debug_filename = save_debug_image(
            image.copy(),
            depth,
            all_objects,
            free_frac,
            obs_frac,
            mode,
        )

    total_time = time.time() - t_start
    app.logger.info(
        "[analyze %s] done, yolo=%.3fs, free_space=%.3fs, objects=%d, has_obstacles=%s, total_time=%.3fs",
        request_id,
        yolo_time,
        free_frac,
        len(all_objects),
        has_obstacles,
        total_time,
    )

    speech = build_speech_text(has_obstacles, all_objects, mode)

    result = {
        "speech": speech,
        "has_obstacles": bool(has_obstacles),
        "objects": all_objects,
        "mode": mode,
        "request_id": request_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    if debug_flag and debug_filename is not None:
        result["debug_image"] = debug_filename

    return jsonify(result)


@app.route("/debug_last")
def debug_last():
    last = get_last_debug_image()
    if last is None:
        return "No debug images yet", 404
    return send_from_directory(DEBUG_DIR, last)


@app.route("/debug_file/<path:filename>")
def debug_file(filename):
    path = os.path.join(DEBUG_DIR, filename)
    if not os.path.isfile(path):
        abort(404)
    return send_from_directory(DEBUG_DIR, filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
