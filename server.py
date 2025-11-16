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

# YOLO: какие классы нам интересны (остальные игнорим)
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
depth_model = AutoModelForDepthEstimation.from_pretrained(
    "Intel/zoedepth-nyu-kitti"
).to(DEVICE)
depth_model.eval()

print("[init] Loading YOLOv8x (COCO)...")
yolo_model = YOLO("yolov8x.pt")
yolo_model.to(DEVICE)


# ----------------------------
#  Вспомогательные функции
# ----------------------------

def current_ts_for_filename() -> str:
    # Да, deprecated, но для нас ок
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

    # В разных версиях HF либо outputs.predicted_depth, либо просто тензор
    if hasattr(outputs, "predicted_depth"):
        depth = outputs.predicted_depth
    else:
        depth = outputs

    # Приводим к (H, W)
    if depth.ndim == 4:
        depth = depth[0, 0]
    elif depth.ndim == 3:
        depth = depth[0]
    else:
        depth = depth.squeeze()

    orig_w, orig_h = image_pil.size
    depth = depth.unsqueeze(0).unsqueeze(0)  # (1, 1, h, w)

    depth_resized = F.interpolate(
        depth,
        size=(orig_h, orig_w),
        mode="bilinear",
        align_corners=False,
    )[0, 0]

    depth_np = depth_resized.detach().cpu().numpy().astype("float32")

    t1 = time.time()
    app.logger.info("[depth] zoedepth forward+resize %.3f s", t1 - t0)

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

    # грубая оценка свободного/занятого пространства
    unique_ids = np.unique(seg)
    for class_id in unique_ids:
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
    if depth is None:
        return None

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

    if boxes is None or num_boxes == 0:
        app.logger.info("[yolo] filtered objects: 0")
        return [], t1 - t0

    h, w = depth.shape
    objects: list[dict] = []

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
                "bbox": [
                    int(round(x1)),
                    int(round(y1)),
                    int(round(x2)),
                    int(round(y2)),
                ],
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
        if nearest is None or d < nearest.get("distance_m", 1e9):
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


def depth_to_debug_image(depth: np.ndarray) -> Image.Image:
    """
    Простейшая псевдоцветная/градационная визуализация глубины.
    Чем ближе — тем ярче.
    """
    h, w = depth.shape
    depth = np.array(depth, copy=True)

    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        # просто чёрное изображение
        return Image.new("RGB", (w, h), (0, 0, 0))

    vmin = float(np.percentile(depth[valid], 5))
    vmax = float(np.percentile(depth[valid], 95))
    if vmax <= vmin:
        vmax = vmin + 1e-6

    depth_clipped = np.clip(depth, vmin, vmax)
    norm = (depth_clipped - vmin) / (vmax - vmin)

    # ближе -> ярче
    norm = 1.0 - norm
    norm = np.clip(norm, 0.0, 1.0)
    img_gray = (norm * 255.0).astype("uint8")

    img = Image.fromarray(img_gray, mode="L").convert("RGB")

    # подпишем, что это depth
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    text = "Depth map"
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.rectangle([0, 0, tw + 10, th + 10], fill=(0, 0, 0))
    draw.text((5, 5), text, font=font, fill=(255, 255, 255))

    return img


def save_debug_images(
    img: Image.Image,
    depth: np.ndarray | None,
    objects: list[dict],
    free_frac: float,
    obs_frac: float,
    mode: str,
) -> tuple[str, str | None]:
    """
    Сохраняем две debug-картинки:
      - debug_<ts>_yolo.jpg  (оригинал + боксы + инфо)
      - debug_<ts>_depth.jpg (визуализация глубины)
    Возвращаем (yolo_filename, depth_filename_or_None)
    """
    ts = current_ts_for_filename()

    # --- YOLO debug image ---
    img_yolo = img.convert("RGB")
    draw = ImageDraw.Draw(img_yolo)

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

    # Общая инфа
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

    info_bbox = ImageDraw.Draw(img_yolo).multiline_textbbox((0, 0), info_text, font=font_info)
    iw = info_bbox[2] - info_bbox[0]
    ih = info_bbox[3] - info_bbox[1]

    draw.rectangle([0, 0, iw + 10, ih + 10], fill=(0, 0, 0, 160))
    draw.multiline_text((5, 5), info_text, font=font_info, fill=(255, 255, 255))

    yolo_filename = f"debug_{ts}_yolo.jpg"
    yolo_path = os.path.join(DEBUG_DIR, yolo_filename)
    img_yolo.save(yolo_path, "JPEG", quality=90)

    # --- Depth debug image ---
    depth_filename = None
    if depth is not None:
        depth_img = depth_to_debug_image(depth)
        depth_draw = ImageDraw.Draw(depth_img)
        try:
            font_info2 = ImageFont.truetype("DejaVuSans.ttf", 18)
        except Exception:
            font_info2 = ImageFont.load_default()
        info2 = f"mode: {mode}\nfree: {free_frac:.3f}\nobs: {obs_frac:.3f}"
        bbox2 = depth_draw.multiline_textbbox((0, 0), info2, font=font_info2)
        iw2 = bbox2[2] - bbox2[0]
        ih2 = bbox2[3] - bbox2[1]
        depth_draw.rectangle([0, 0, iw2 + 10, ih2 + 10], fill=(0, 0, 0, 160))
        depth_draw.multiline_text((5, 5), info2, font=font_info2, fill=(255, 255, 255))

        depth_filename = f"debug_{ts}_depth.jpg"
        depth_path = os.path.join(DEBUG_DIR, depth_filename)
        depth_img.save(depth_path, "JPEG", quality=90)

    return yolo_filename, depth_filename


def get_last_debug_yolo_image() -> str | None:
    files = [
        f
        for f in os.listdir(DEBUG_DIR)
        if f.lower().endswith("_yolo.jpg") and f.startswith("debug_")
    ]
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
    raw = request.get_data()
    t1 = time.time()

    image = None
    image_bytes_for_log = raw

    # 1) Пробуем как "сырые байты" (как раньше)
    if raw:
        try:
            image = Image.open(io.BytesIO(raw))
        except Exception:
            image = None

    # 2) multipart/form-data
    if image is None and request.files:
        try:
            f = next(iter(request.files.values()))
            fb = f.read()
            image = Image.open(io.BytesIO(fb))
            image_bytes_for_log = fb
        except Exception:
            image = None

    # 3) JSON с base64
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
        # total_time мы логируем позже
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
    debug_yolo_filename = None
    debug_depth_filename = None
    if debug_flag:
        debug_yolo_filename, debug_depth_filename = save_debug_images(
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

    if debug_flag and debug_yolo_filename is not None:
        # для приложения: одно название, как раньше
        result["debug_image"] = debug_yolo_filename

    return jsonify(result)


@app.route("/debug_last")
def debug_last():
    """
    Старый эндпоинт — возвращаем последнюю YOLO-картинку.
    """
    last = get_last_debug_yolo_image()
    if last is None:
        return "No debug images yet", 404
    return send_from_directory(DEBUG_DIR, last)


@app.route("/debug_file/<path:filename>")
def debug_file(filename):
    path = os.path.join(DEBUG_DIR, filename)
    if not os.path.isfile(path):
        abort(404)
    return send_from_directory(DEBUG_DIR, filename)


@app.route("/debug")
def debug_page():
    """
    Новая страница:
      - показывает последние 50 наборов картинок
      - у каждого набора: слева YOLO, справа depth (если есть)
    """
    files = [
        f
        for f in os.listdir(DEBUG_DIR)
        if f.lower().endswith("_yolo.jpg") and f.startswith("debug_")
    ]
    if not files:
        return "<h1>No debug images yet</h1>", 200

    files.sort()  # по имени → по времени
    files = files[-50:]  # последние 50

    items_html = []
    for fname in reversed(files):  # новые сверху
        # fname = "debug_<ts>_yolo.jpg"
        ts = fname[len("debug_") : -len("_yolo.jpg")]
        yolo_name = fname
        depth_name = f"debug_{ts}_depth.jpg"
        depth_exists = os.path.isfile(os.path.join(DEBUG_DIR, depth_name))
        ts_display = ts

        row = [
            '<div class="item">',
            f'<div class="meta">debug id: {ts_display}</div>',
            '<div class="imgs">',
            f'<div class="img-block"><div class="label">YOLO</div><img src="/debug_file/'
            f'{yolo_name}" loading="lazy"></div>',
        ]
        if depth_exists:
            row.append(
                f'<div class="img-block"><div class="label">Depth</div>'
                f'<img src="/debug_file/{depth_name}" loading="lazy"></div>'
            )
        row.append("</div></div>")
        items_html.append("\n".join(row))

    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="UTF-8">
  <title>VisionGuide Debug</title>
  <style>
    body {{
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #111;
      color: #eee;
      margin: 0;
      padding: 0;
    }}
    h1 {{
      text-align: center;
      padding: 16px;
      margin: 0;
      background: #181818;
      border-bottom: 1px solid #333;
    }}
    .container {{
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }}
    .item {{
      background: #181818;
      border-radius: 8px;
      padding: 12px;
      border: 1px solid #333;
    }}
    .meta {{
      font-size: 12px;
      color: #aaa;
      margin-bottom: 8px;
    }}
    .imgs {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
    }}
    .img-block {{
      display: flex;
      flex-direction: column;
      gap: 4px;
    }}
    .img-block .label {{
      font-size: 13px;
      color: #ccc;
    }}
    img {{
      max-width: 360px;
      border-radius: 6px;
      border: 1px solid #444;
    }}
  </style>
</head>
<body>
  <h1>VisionGuide Debug (last {len(files)} frames)</h1>
  <div class="container">
    {''.join(items_html)}
  </div>
</body>
</html>
"""
    return html


if __name__ == "__main__":
    # Для локального запуска; на сервере у тебя systemd + start_server.sh
    app.run(host="0.0.0.0", port=8000)
