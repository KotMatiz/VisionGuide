import io
import os
import time
import json
import base64
from datetime import datetime

from flask import Flask, request, jsonify, send_from_directory, abort

import numpy as np
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError

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

# Максимальный размер тела запроса (байт), чтобы не убить сервер
MAX_REQUEST_SIZE = 5 * 1024 * 1024  # 5 МБ

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
depth_model = AutoModelForDepthEstimation.from_pretrained(
    "Intel/zoedepth-nyu-kitti"
).to(DEVICE)
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


def run_zoedepth(image_pil: Image.Image) -> np.ndarray | None:
    """
    Запускаем ZoeDepth и возвращаем depth-карту (H, W) в float32.
    Если что-то пошло не так — логируем и возвращаем None.
    """
    t0 = time.time()
    try:
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
        app.logger.info("[depth] zoedepth forward+resize %.3f s", t1 - t0)

        return depth_np
    except Exception:
        app.logger.exception("[depth] ZoeDepth failed")
        return None


def run_segformer(
    image_pil: Image.Image,
) -> tuple[np.ndarray | None, float, float, int, float, float]:
    """
    Запускаем SegFormer, возвращаем:
      seg_mask: (H, W) int32 – id класса для каждого пикселя (или None при ошибке)
      free_frac: доля пикселей "проходимых" классов по всему кадру
      obs_frac: доля "непроходимых" по всему кадру
      num_pixels: всего пикселей
      free_bottom_frac: доля проходимых в нижней части кадра
      obs_bottom_frac: доля непр. в нижней части кадра
    """
    t0 = time.time()
    try:
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

        # по всему кадру
        for class_id in np.unique(seg):
            label = seg_id2label.get(int(class_id), "")
            mask = seg == class_id
            cnt = int(mask.sum())
            if label in FREE_LABELS:
                free_count += cnt
            else:
                obs_count += cnt

        free_frac = free_count / max(1, num_pixels)
        obs_frac = obs_count / max(1, num_pixels)

        # нижняя часть кадра (нижние 40%)
        bottom_start = int(h * 0.6)
        bottom_mask = np.zeros_like(seg, dtype=bool)
        bottom_mask[bottom_start:] = True

        free_bottom = 0
        obs_bottom = 0
        total_bottom = int(bottom_mask.sum())

        if total_bottom > 0:
            for class_id in np.unique(seg):
                label = seg_id2label.get(int(class_id), "")
                mask = (seg == class_id) & bottom_mask
                cnt = int(mask.sum())
                if not cnt:
                    continue
                if label in FREE_LABELS:
                    free_bottom += cnt
                else:
                    obs_bottom += cnt

            free_bottom_frac = free_bottom / total_bottom
            obs_bottom_frac = obs_bottom / total_bottom
        else:
            free_bottom_frac = 1.0
            obs_bottom_frac = 0.0

        return seg, free_frac, obs_frac, num_pixels, free_bottom_frac, obs_bottom_frac

    except Exception:
        app.logger.exception("[segformer] failed")
        return None, 0.0, 1.0, 0, 1.0, 0.0


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


def estimate_distance_from_depth(
    depth: np.ndarray | None, bbox
) -> float | None:
    """
    depth: (H, W) float32 или None
    bbox: [x1, y1, x2, y2]
    Возвращаем медиану глубины внутри бокса.
    """
    if depth is None or depth.ndim != 2:
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


def run_yolo(image_pil: Image.Image, depth: np.ndarray | None):
    """
    Запускаем YOLOv8x, возвращаем список объектов в нашем формате.
    При любой ошибке возвращаем пустой список.
    """
    t0 = time.time()

    try:
        img_np = pil_to_numpy_rgb(image_pil)
        results = yolo_model(img_np, verbose=False)[0]

        t1 = time.time()
        boxes = results.boxes
        num_boxes = boxes.xyxy.shape[0] if boxes is not None else 0
        app.logger.info("[yolo] forward %.3f s, boxes=%d", t1 - t0, num_boxes)

        if boxes is None or num_boxes == 0:
            app.logger.info("[yolo] filtered objects: 0")
            return [], t1 - t0

        h, w, _ = img_np.shape
        objects = []

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

    except Exception:
        app.logger.exception("[yolo] failed")
        t1 = time.time()
        return [], t1 - t0


def build_speech_text(has_obstacles, objects, mode: str) -> str:
    """
    Формируем короткую фразу для озвучки.
    Формат как раньше: "Впереди стол, в полутора метрах".
    Без "чуть левее/правее".
    """

    if not has_obstacles:
        if mode == "outdoor":
            return "Впереди свободно"
        else:
            return "Впереди нет препятствий"

    if not objects:
        if mode == "outdoor":
            return "Возможно препятствие впереди"
        else:
            return "Возможно препятствие, низ кадра закрыт"

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


def save_debug_images(
    img: Image.Image,
    depth: np.ndarray | None,
    objects: list[dict],
    free_frac: float,
    obs_frac: float,
    mode: str,
    ts: str,
) -> tuple[str | None, str | None]:
    """
    Сохраняем две debug-картинки:
      - с боксами YOLO: debug_<ts>_yolo.jpg
      - с глубиной:    debug_<ts>_depth.jpg
    Возвращаем (yolo_filename, depth_filename).
    """
    yolo_fn = None
    depth_fn = None

    # 1) YOLO-боксы
    try:
        img_yolo = img.convert("RGB")
        draw = ImageDraw.Draw(img_yolo)

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

        yolo_fn = f"debug_{ts}_yolo.jpg"
        path = os.path.join(DEBUG_DIR, yolo_fn)
        img_yolo.save(path, "JPEG", quality=90)
    except Exception:
        app.logger.exception("[debug] failed to save yolo debug image")
        yolo_fn = None

    # 2) Depth
    try:
        if depth is not None and depth.ndim == 2:
            d = depth.copy()
            d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
            if d.max() > 0:
                d = d / d.max()
            d_img = (d * 255).astype(np.uint8)
            d_img = Image.fromarray(d_img)
            d_img = d_img.resize(img.size, Image.BILINEAR)
            d_img = d_img.convert("L").convert("RGB")
            draw_d = ImageDraw.Draw(d_img)
            try:
                font_d = ImageFont.truetype("DejaVuSans.ttf", 18)
            except Exception:
                font_d = ImageFont.load_default()
            draw_d.text(
                (10, 10),
                "Depth map",
                font=font_d,
                fill=(255, 255, 255),
            )
            depth_fn = f"debug_{ts}_depth.jpg"
            path_d = os.path.join(DEBUG_DIR, depth_fn)
            d_img.save(path_d, "JPEG", quality=90)
        else:
            no_d = img.convert("RGB")
            draw_nd = ImageDraw.Draw(no_d)
            try:
                font_nd = ImageFont.truetype("DejaVuSans.ttf", 18)
            except Exception:
                font_nd = ImageFont.load_default()
            draw_nd.text(
                (10, 10),
                "Depth not available",
                font=font_nd,
                fill=(255, 0, 0),
            )
            depth_fn = f"debug_{ts}_depth.jpg"
            path_nd = os.path.join(DEBUG_DIR, depth_fn)
            no_d.save(path_nd, "JPEG", quality=90)
    except Exception:
        app.logger.exception("[debug] failed to save depth debug image")
        depth_fn = None

    return yolo_fn, depth_fn


def list_debug_pairs(limit: int = 50):
    """
    Ищем файлы debug_<ts>_yolo.jpg и debug_<ts>_depth.jpg,
    плюс debug_<ts>.json с мета-информацией.
    """
    try:
        all_files = os.listdir(DEBUG_DIR)
    except Exception:
        app.logger.exception("[debug] failed to list files")
        return []

    image_files = [
        f for f in all_files
        if f.startswith("debug_") and f.lower().endswith(".jpg")
    ]
    yolo_files = [f for f in image_files if "_yolo" in f]
    yolo_files.sort(reverse=True)

    pairs = []
    for f in yolo_files:
        if len(pairs) >= limit:
            break
        # debug_<ts>_yolo.jpg
        ts_part = f[len("debug_") :].split("_yolo")[0]
        depth_name = f"debug_{ts_part}_depth.jpg"
        depth_exists = depth_name in image_files

        meta = None
        json_name = f"debug_{ts_part}.json"
        json_path = os.path.join(DEBUG_DIR, json_name)
        if os.path.isfile(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as jf:
                    meta = json.load(jf)
            except Exception:
                meta = None

        pairs.append(
            {
                "timestamp": ts_part,
                "yolo": f,
                "depth": depth_name if depth_exists else None,
                "meta": meta,
            }
        )

    return pairs


def get_last_debug_image() -> str | None:
    """
    Для /debug_last — просто самое новое yolo-изображение.
    """
    pairs = list_debug_pairs(limit=1)
    if not pairs:
        return None
    return pairs[0]["yolo"]


# ----------------------------
#  Маршруты
# ----------------------------

@app.route("/analyze", methods=["POST"])
def analyze():
    t_start = time.time()
    request_id = str(int(t_start * 1000))

    mode = request.args.get("mode", "indoor")
    debug_flag = request.args.get("debug", "0") in ("1", "true", "True")

    try:
        # --- читаем body ---
        t0 = time.time()
        raw = request.get_data()
        t1 = time.time()

        if not raw:
            app.logger.warning("[analyze %s] empty body", request_id)
            return jsonify({"error": "empty_body"}), 400

        if len(raw) > MAX_REQUEST_SIZE:
            app.logger.warning(
                "[analyze %s] body too large: %d bytes", request_id, len(raw)
            )
            return jsonify({"error": "body_too_large"}), 400

        image = None
        image_bytes_for_log = raw

        # 1) сырые байты — сразу картинка
        try:
            image = Image.open(io.BytesIO(raw))
        except UnidentifiedImageError:
            image = None
        except Exception:
            image = None

        # 2) multipart/form-data
        if image is None and request.files:
            try:
                f = next(iter(request.files.values()))
                fb = f.read()
                image = Image.open(io.BytesIO(fb))
                image_bytes_for_log = fb
            except UnidentifiedImageError:
                image = None
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
            except (json.JSONDecodeError, KeyError, ValueError, UnidentifiedImageError):
                image = None
            except Exception:
                image = None

        if image is None:
            app.logger.error("[analyze %s] failed to open image", request_id)
            return jsonify({"error": "invalid_image"}), 400

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
        (
            seg,
            free_frac,
            obs_frac,
            num_pixels,
            free_bottom_frac,
            obs_bottom_frac,
        ) = run_segformer(image)

        # --- YOLO ---
        yolo_objects, yolo_time = run_yolo(image, depth)
        all_objects = yolo_objects

        # --- Логика has_obstacles ---

        real_obstacles = []
        for obj in all_objects:
            d = obj.get("distance_m")
            frac = obj.get("fraction", 0.0)
            if d is not None and d > 0 and d < 2.5 and frac >= 0.01:
                real_obstacles.append(obj)

        if real_obstacles:
            has_obstacles = True
        else:
            if free_bottom_frac < 0.20 and obs_bottom_frac > 0.40:
                has_obstacles = True
            else:
                has_obstacles = False

        speech = build_speech_text(has_obstacles, all_objects, mode)

        # --- Debug (картинки + JSON) ---
        debug_yolo_filename = None
        debug_depth_filename = None
        ts = None
        if debug_flag:
            ts = current_ts_for_filename()
            debug_yolo_filename, debug_depth_filename = save_debug_images(
                image.copy(),
                depth,
                all_objects,
                free_frac,
                obs_frac,
                mode,
                ts,
            )
            # JSON с ответом сервера
            try:
                debug_meta = {
                    "timestamp": ts,
                    "request_id": request_id,
                    "mode": mode,
                    "speech": speech,
                    "has_obstacles": bool(has_obstacles),
                    "free_frac": float(free_frac),
                    "obs_frac": float(obs_frac),
                    "free_bottom_frac": float(free_bottom_frac),
                    "obs_bottom_frac": float(obs_bottom_frac),
                    "objects": all_objects,
                }
                json_path = os.path.join(DEBUG_DIR, f"debug_{ts}.json")
                with open(json_path, "w", encoding="utf-8") as jf:
                    json.dump(debug_meta, jf, ensure_ascii=False, indent=2)
            except Exception:
                app.logger.exception("[debug] failed to save debug json")

        total_time = time.time() - t_start
        app.logger.info(
            "[analyze %s] done, yolo=%.3fs, free=%.3f, free_bottom=%.3f, objects=%d, has_obstacles=%s, total_time=%.3fs",
            request_id,
            yolo_time,
            free_frac,
            free_bottom_frac,
            len(all_objects),
            has_obstacles,
            total_time,
        )

        result = {
            "speech": speech,
            "has_obstacles": bool(has_obstacles),
            "objects": all_objects,
            "mode": mode,
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

        if debug_flag:
            result["debug_yolo_image"] = debug_yolo_filename
            result["debug_depth_image"] = debug_depth_filename

        return jsonify(result)

    except Exception:
        app.logger.exception("[analyze %s] unexpected error", request_id)
        return jsonify({"error": "internal_error"}), 500


@app.route("/debug_last")
def debug_last():
    """
    Отдать последнее YOLO-debug изображение (как раньше).
    """
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


@app.route("/debug")
def debug_page():
    """
    Страница: последние 50 кадров, у каждого:
      - debug_*_yolo.jpg
      - debug_*_depth.jpg (если есть)
      - текст: has_obstacles + speech + объекты
    """
    pairs = list_debug_pairs(limit=50)

    html_parts = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8' />",
        "<title>VisionGuide Debug</title>",
        "<style>",
        "body { font-family: sans-serif; padding: 16px; background: #111; color: #eee; }",
        "h1 { margin-top: 0; }",
        ".grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 16px; }",
        ".card { background: #222; border-radius: 8px; padding: 10px; }",
        ".card h3 { margin: 4px 0 8px 0; font-size: 14px; color: #aaa; }",
        ".imgs { display: flex; gap: 8px; margin-bottom: 8px; }",
        ".imgs img { max-width: 100%; border-radius: 4px; }",
        "ul { margin: 4px 0 0 16px; padding: 0; }",
        "li { font-size: 13px; margin-bottom: 2px; }",
        "</style>",
        "</head><body>",
        "<h1>VisionGuide Debug</h1>",
        "<p>Последние кадры: YOLO + Depth + ответ сервера.</p>",
        "<div class='grid'>",
    ]

    if not pairs:
        html_parts.append("<p>Пока нет debug-изображений.</p>")
    else:
        for p in pairs:
            ts = p["timestamp"]
            yolo = p["yolo"]
            depth = p["depth"]
            meta = p.get("meta") or {}

            has_obs = meta.get("has_obstacles")
            speech = meta.get("speech")
            objs = meta.get("objects", [])
            mode = meta.get("mode", "")

            html_parts.append("<div class='card'>")
            html_parts.append(f"<h3>{ts} ({mode})</h3>")
            html_parts.append("<div class='imgs'>")
            if yolo:
                html_parts.append(
                    f"<div><div>YOLO</div><img src='/debug_file/{yolo}' loading='lazy' /></div>"
                )
            if depth:
                html_parts.append(
                    f"<div><div>Depth</div><img src='/debug_file/{depth}' loading='lazy' /></div>"
                )
            html_parts.append("</div>")  # .imgs

            # Текстовая часть
            if meta:
                if has_obs is not None:
                    text = "Да" if has_obs else "Нет"
                    color = "#f55" if has_obs else "#5f5"
                    html_parts.append(
                        f"<div>Препятствие: <span style='color:{color}'>{text}</span></div>"
                    )
                if speech:
                    html_parts.append(
                        f"<div>Ответ: <b>{speech}</b></div>"
                    )
                if objs:
                    html_parts.append("<div>Объекты:</div><ul>")
                    for obj in objs[:5]:
                        lab = obj.get("label", "объект")
                        direction = obj.get("direction", "")
                        dist = obj.get("distance_m")
                        if dist is not None:
                            dist_str = f"{dist:.1f}м"
                        else:
                            dist_str = "?"
                        if direction:
                            html_parts.append(
                                f"<li>{lab} ({direction}), {dist_str}</li>"
                            )
                        else:
                            html_parts.append(
                                f"<li>{lab}, {dist_str}</li>"
                            )
                    if len(objs) > 5:
                        html_parts.append(
                            f"<li>... ещё {len(objs) - 5}</li>"
                        )
                    html_parts.append("</ul>")
            else:
                html_parts.append("<div>Нет метаданных (JSON не найден).</div>")

            html_parts.append("</div>")  # .card

    html_parts.append("</div></body></html>")
    return "\n".join(html_parts)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
