"""Fine-tuna um YOLO de segmentação para a classe única "food" no FoodSeg103.

Uso:
    uv run python scripts/train.py                       # treino completo (GPU)
    uv run python scripts/train.py --epochs 2 --fraction 0.05  # teste rápido
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT / "data" / "foodseg103" / "data.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yolov8n-seg.pt",
                        help="checkpoint base (yolov8n-seg.pt é leve e cabe folgado em 8GB)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--fraction", type=float, default=1.0,
                        help="fração do dataset de treino a usar (para testes rápidos)")
    args = parser.parse_args()

    if not DATA_YAML.exists():
        raise SystemExit("Dataset não encontrado. Rode antes: uv run python scripts/download_dataset.py")

    model = YOLO(args.model)
    model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        fraction=args.fraction,
        device=0,
        project=str(ROOT / "runs"),
        name="food-seg",
        exist_ok=True,
    )
    print(f"Treino concluído. Melhor peso: {ROOT / 'runs' / 'food-seg' / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
