#Подсчет товаров на полке

Учебная работа: разработка системы автоматического обнаружения и подсчёта товаров на изображениях торговых полок и сравнительный анализ пяти архитектур нейронных сетей.
 
## Постановка задачи
 
Дано фото торговой полки — нужно обнаружить все товары, посчитать их количество и подсветить ошибки (пропуски и ложные срабатывания). Задача относится к классу **dense object detection** — на одном изображении может быть до 400+ товаров с сильными перекрытиями.
 
## Датасет
 
[SKU-110K](https://github.com/eg4000/SKU110K_CVPR19) — 11 743 изображения, 1,73 млн bounding box, один класс «product». В работе использовано стратифицированное подмножество:
- 3 000 train / 400 val / 1 000 test
- random seed 42
## Сравниваемые архитектуры
 
| Модель | Семейство | Фреймворк |
|---|---|---|
| YOLOv8s | Anchor-free one-stage CNN | Ultralytics |
| RT-DETR-L | End-to-end Transformer | Ultralytics |
| Faster R-CNN R50 FPN v2 | Two-stage CNN | torchvision |
| RetinaNet R50 FPN v2 | Focal-loss one-stage CNN | torchvision |
| EfficientDet-D0 | Compound-scaled CNN | effdet |
 
Все модели обучались в единой конфигурации: 20 эпох, разрешение входа 640×640, GPU NVIDIA A100.
 
## Результаты
 
| Модель | mAP@0.5:0.95 | MAE подсчёта | FPS | Размер, МБ |
|---|---|---|---|---|
| **YOLOv8s** ★ | **0,4107** | **16,31** | 20,2 | 22,5 |
| RT-DETR-L | 0,4090 | 47,17 | 12,0 | 66,2 |
| Faster R-CNN R50 | 0,3835 | 48,54 | 36,4 | 173,4 |
| EfficientDet-D0 | 0,3547 | 49,33 | 30,5 | 17,0 |
| RetinaNet R50 | 0,3293 | 25,19 | 39,7 | 145,7 |
 
★ Итоговая модель — **YOLOv8s** (лидер по mAP и по точности подсчёта при компактном размере).
 
## Демонстрационное приложение
 
Веб-приложение на базе Gradio с двумя режимами работы:
- **Режим инференса** — загрузка произвольного фото, детекция и подсчёт товаров
- **Режим оценки** — сравнение предсказаний модели с эталонной разметкой, визуализация TP/FP/FN
Запуск в Google Colab — выполнить `app.py` или соответствующую ячейку в ноутбуке.
 
## Состав репозитория
 
```
├── README.md                — описание проекта
├── app.py                   — скрипт Gradio-приложения
├── shelf_detection.ipynb    — Colab-ноутбук (подготовка данных,
│                              обучение 5 моделей, оценка, визуализация)
└── requirements.txt         — версии библиотек
```
 
## Данные и веса моделей
 
Веса обученных моделей, промежуточные результаты и визуализации доступны на Google Drive:
 
**[Скачать данные проекта](https://drive.google.com/drive/folders/18uTZJH7fVBYKQNPtcQRezJ6r8nglYm5l?usp=sharing)**
 
Структура папки на Google Drive:
```
shelf_detection/
├── models/
│   ├── yolov8s/weights/best.pt        — веса YOLOv8s (22,5 МБ)
│   ├── rtdetr_l/weights/best.pt       — веса RT-DETR-L (66,2 МБ)
│   ├── faster_rcnn_r50/best.pt        — веса Faster R-CNN R50 (173,4 МБ)
│   ├── retinanet_r50/best.pt          — веса RetinaNet R50 (145,7 МБ)
│   └── efficientdet_d0/best.pt        — веса EfficientDet-D0 (17,0 МБ)
├── data/subset/
│   ├── train.csv                      — разбиение обучающей выборки
│   ├── val.csv                        — разбиение валидационной выборки
│   └── test.csv                       — разбиение тестовой выборки
└── results/
    ├── comparison.csv                 — сводная таблица метрик
    ├── preds_yolov8s.pkl              — предсказания YOLOv8s
    ├── preds_rtdetr.pkl               — предсказания RT-DETR-L
    ├── preds_frcnn.pkl                — предсказания Faster R-CNN
    ├── preds_retina.pkl               — предсказания RetinaNet
    ├── preds_eff.pkl                  — предсказания EfficientDet-D0
    └── visualizations/                — визуализации сравнения моделей
```
 m
## Воспроизведение
 
1. Открыть `shelf_detection.ipynb` в Google Colab (нужен GPU, рекомендуется A100)
2. Подключить Google Drive
3. Скачать датасет SKU-110K (выполняется автоматически в ноутбуке)
4. Выполнить ячейки последовательно
## Технологии
 
Python 3.12, PyTorch 2.11, Ultralytics 8.4.75, torchvision 0.18, effdet 0.4, pycocotools, Gradio.
 
## Авторы
 
Петчак М.И., группа УБВТ2304, 2026.
