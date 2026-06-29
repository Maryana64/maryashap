from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np, time, os, gradio as gr

def compute_iou_matrix(boxes_a, boxes_b):
    x1 = np.maximum(boxes_a[:, 0:1], boxes_b[:, 0])
    y1 = np.maximum(boxes_a[:, 1:2], boxes_b[:, 1])
    x2 = np.minimum(boxes_a[:, 2:3], boxes_b[:, 2])
    y2 = np.minimum(boxes_a[:, 3:4], boxes_b[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-6)

class ShelfProductDetector:
    def __init__(self):
        self.model = YOLO('/content/drive/MyDrive/shelf_detection/models/yolov8s/weights/best.pt')
        self.model_name = 'YOLOv8s'
    def detect(self, image, conf_thresh=0.25):
        t0 = time.time()
        results = self.model(image, conf=conf_thresh, imgsz=640, verbose=False, max_det=300)
        ms = (time.time() - t0) * 1000
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return np.zeros((0, 4)), np.zeros(0), 0, ms
        return results[0].boxes.xyxy.cpu().numpy(), results[0].boxes.conf.cpu().numpy(), len(results[0].boxes), ms
    def draw(self, image, boxes, count):
        img = image.copy(); draw = ImageDraw.Draw(img)
        for box in boxes:
            draw.rectangle([box[0], box[1], box[2], box[3]], outline='lime', width=2)
        try: font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 28)
        except: font = ImageFont.load_default()
        text = f'Найдено товаров: {count}'
        bbox = draw.textbbox((10, 10), text, font=font)
        draw.rectangle([bbox[0]-5, bbox[1]-5, bbox[2]+5, bbox[3]+5], fill='black')
        draw.text((10, 10), text, fill='lime', font=font)
        return img
    def evaluate(self, image, boxes, scores, gt_boxes):
        img = image.copy(); draw = ImageDraw.Draw(img)
        tp_boxes, fp_boxes, matched_gt = [], [], set()
        if len(boxes) > 0 and len(gt_boxes) > 0:
            iou_mat = compute_iou_matrix(boxes, gt_boxes)
            for pi in np.argsort(-scores):
                best_gt, best_iou = -1, 0.5
                for gi in range(len(gt_boxes)):
                    if gi not in matched_gt and iou_mat[pi, gi] > best_iou:
                        best_iou = iou_mat[pi, gi]; best_gt = gi
                if best_gt >= 0: tp_boxes.append(boxes[pi]); matched_gt.add(best_gt)
                else: fp_boxes.append(boxes[pi])
        else: fp_boxes = list(boxes)
        fn_boxes = [gt_boxes[gi] for gi in range(len(gt_boxes)) if gi not in matched_gt]
        for box in tp_boxes: draw.rectangle([box[0],box[1],box[2],box[3]], outline='lime', width=2)
        for box in fp_boxes: draw.rectangle([box[0],box[1],box[2],box[3]], outline='red', width=2)
        for box in fn_boxes: draw.rectangle([box[0],box[1],box[2],box[3]], outline='yellow', width=3)
        try: font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 22)
        except: font = ImageFont.load_default()
        acc = len(tp_boxes) / max(len(gt_boxes), 1) * 100
        y = 10
        for line in [f'Эталон: {len(gt_boxes)}  Найдено: {len(boxes)}',
                     f'TP={len(tp_boxes)}  FP={len(fp_boxes)}  FN={len(fn_boxes)}',
                     f'Точность подсчёта: {acc:.1f}%']:
            bbox = draw.textbbox((10, y), line, font=font)
            draw.rectangle([bbox[0]-5, bbox[1]-2, bbox[2]+5, bbox[3]+2], fill='black')
            draw.text((10, y), line, fill='white', font=font); y += 30
        return img, len(tp_boxes), len(fp_boxes), len(fn_boxes), acc

detector = ShelfProductDetector()
test_files = sorted(os.listdir('/content/yolo_dataset/images/test'))

def load_gt(img_name):
    path = f'/content/yolo_dataset/labels/test/{img_name[:-4]}.txt'
    if not os.path.exists(path): return np.zeros((0, 4))
    boxes = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5: continue
            _, cx, cy, bw, bh = map(float, parts[:5])
            boxes.append([(cx-bw/2)*640, (cy-bh/2)*640, (cx+bw/2)*640, (cy+bh/2)*640])
    return np.array(boxes) if boxes else np.zeros((0, 4))

def mode_inference(image, conf):
    if image is None: return None, "Загрузите изображение"
    pil = image.resize((640, 640), Image.BILINEAR)
    boxes, scores, count, ms = detector.detect(pil, conf)
    return detector.draw(pil, boxes, count), (
        f'**Найдено товаров:** {count}\n\n**Время инференса:** {ms:.1f} мс\n\n'
        f'**Модель:** {detector.model_name}\n\n**Порог уверенности:** {conf}')

def mode_eval(img_name, conf):
    if not img_name: return None, "Выберите файл"
    pil = Image.open(f'/content/yolo_dataset/images/test/{img_name}').convert('RGB').resize((640,640), Image.BILINEAR)
    gt = load_gt(img_name)
    boxes, scores, count, ms = detector.detect(pil, conf)
    img, tp, fp, fn, acc = detector.evaluate(pil, boxes, scores, gt)
    return img, (f'**Файл:** {img_name}\n\n**Эталон:** {len(gt)}\n\n**Найдено:** {count}\n\n'
                 f'**TP:** {tp}  **FP:** {fp}  **FN:** {fn}\n\n**Точность:** {acc:.1f}%\n\n**Время:** {ms:.1f} мс')

# ===== ТЕМА: поменяй на gr.themes.Default() для тёмной (первый отчёт) =====
with gr.Blocks(title='Подсчёт товаров на полке', theme=gr.themes.Soft()) as demo:
    gr.Markdown('# Система автоматического подсчёта товаров на полке\n'
                'Реализована на основе YOLOv8s — лучшей модели по итогам сравнительного анализа 5 архитектур.')
    with gr.Tab('Режим инференса (произвольное фото)'):
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type='pil', label='Загрузить фото полки')
                conf1 = gr.Slider(0.1, 0.9, value=0.25, step=0.05, label='Порог уверенности')
                gr.Button('Найти и посчитать', variant='primary').click(mode_inference, [inp, conf1], [gr.Image(type='pil', label='Результат'), gr.Markdown()])
    with gr.Tab('Режим оценки (тестовое фото с эталоном)'):
        gr.Markdown('Выберите тестовое изображение из SKU-110K.')
        with gr.Row():
            with gr.Column():
                dd = gr.Dropdown(choices=test_files, label='Тестовое изображение')
                conf2 = gr.Slider(0.1, 0.9, value=0.25, step=0.05, label='Порог уверенности')
                gr.Button('Оценить', variant='primary').click(mode_eval, [dd, conf2], [gr.Image(type='pil', label='Результат с TP/FP/FN'), gr.Markdown()])
    gr.Markdown('---\n🟢 **Зелёный** — TP  🔴 **Красный** — FP  🟡 **Жёлтый** — FN')

demo.launch(share=True, debug=False)