# ggcnn

Актуальная реализация GG-CNN для обучения на Cornell и Jacquard
с отдельными скриптами:
- `scripts/prepare_cornell.py` — конвертация PCD -> depth и генерация индекса
- `scripts/prepare_jacquard.py` — генерация индекса Jacquard
- `scripts/train.py` — обучение (Cornell, Jacquard)
- `scripts/infer_single.py` — инференс на одном кадре
- `scripts/eval.py` — IoU-оценка 

## Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Датасеты

### Cornell Grasping Dataset
Папки `01..10`, на каждый пример:
- depth: `*d.tiff` / `*d.png`
- grasps: `*cpos.txt` 


### Jacquard Dataset
Официальный формат: файл `*_grasps.txt`,
`x;y;theta(deg);opening;jaws size` 

Индекс:
```bash
python scripts/prepare_jacquard.py --dataset-path /path/to/jacquard --output-index /path/to/jacquard_index.json
```

## train

Cornell:
```bash
python scripts/train.py --network ggcnn --dataset cornell --index /path/to/cornell_index.json --outdir output/cornell_run
```

Jacquard:
```bash
python scripts/train.py --network ggcnn2 --dataset jacquard --index /path/to/jacquard_index.json --outdir output/jacquard_run
```

Cornell + Jacquard:
```bash
python scripts/train.py --network ggcnn2 --dataset mixed   --index /path/to/cornell_index.json /path/to/jacquard_index.json   --mix-weights 0.5 0.5   --outdir output/mixed_run
```

## Инференс на одном кадре

```bash
python scripts/infer_single.py --checkpoint output/cornell_run/best.pt --depth /path/to/depth.png --vis out.png
```

output:
- center (x, y) в пикселях
- angle (rad) в координатах изображения (x вправо, y вниз)
- opening width (px)
- score
