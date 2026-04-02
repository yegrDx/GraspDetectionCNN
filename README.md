# ggcnn_modern (PyTorch >= 2.1)

Современная автономная реализация GG-CNN (RSS 2018) для обучения на Cornell и Jacquard
с отдельными скриптами:
- `scripts/prepare_cornell.py` — конвертация PCD -> depth и генерация индекса
- `scripts/prepare_jacquard.py` — генерация индекса Jacquard
- `scripts/train.py` — обучение (Cornell, Jacquard или микс)
- `scripts/infer_single.py` — инференс на одном кадре (depth)
- `scripts/eval.py` — IoU-оценка (опционально)

## Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Датасеты

### Cornell Grasping Dataset
Обычно папки `01..10`, на каждый пример:
- depth: `*d.tiff` / `*d.png` (если есть) **или** исходный `*.pcd`
- grasps: `*cpos.txt` (по 4 строки (x y) на прямоугольник; возможны NaN строки)

Если у вас только `.pcd`, сначала сделайте depth/индекс:
```bash
python scripts/prepare_cornell.py --dataset-path /path/to/cornell --output-index /path/to/cornell_index.json --convert-pcd
```

### Jacquard Dataset
Официальный формат: файл `*_grasps.txt`, каждая строка:
`x;y;theta(deg);opening;jaws size` (все в пикселях, y вниз).  

Индекс:
```bash
python scripts/prepare_jacquard.py --dataset-path /path/to/jacquard --output-index /path/to/jacquard_index.json
```

## Обучение

Cornell:
```bash
python scripts/train.py --network ggcnn --dataset cornell --index /path/to/cornell_index.json --outdir output/cornell_run
```

Jacquard:
```bash
python scripts/train.py --network ggcnn2 --dataset jacquard --index /path/to/jacquard_index.json --outdir output/jacquard_run
```

Cornell + Jacquard (микс):
```bash
python scripts/train.py --network ggcnn2 --dataset mixed   --index /path/to/cornell_index.json /path/to/jacquard_index.json   --mix-weights 0.5 0.5   --outdir output/mixed_run
```

## Инференс на одном кадре

```bash
python scripts/infer_single.py --checkpoint output/cornell_run/best.pt --depth /path/to/depth.png --vis out.png
```

Выводит grasp:
- center (x, y) в пикселях
- angle (rad) в координатах изображения (x вправо, y вниз)
- opening width (px)
- score

## Примечания про симулятор
GG-CNN предсказывает grasp в плоскости изображения. Чтобы применить в симуляторе:
1) переведите (x,y,depth) в 3D через intrinsics камеры,
2) поверните grasp-ориентацию в frame камеры,
3) трансформируйте в base_link/мир (TF/URDF),
4) подберите approach и длину пальцев под вашу модель.
