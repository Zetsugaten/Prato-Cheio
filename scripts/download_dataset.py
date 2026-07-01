"""Baixa o FoodSeg103 do Hugging Face e converte para o formato YOLO-seg.

O FoodSeg103 traz máscaras semânticas com 103 classes de comida (0 = fundo).
Para estimar QUANTIDADE de comida não precisamos distinguir arroz de feijão,
então todas as classes são colapsadas numa única classe binária "food" (id 0).
As máscaras viram polígonos normalizados, que é o formato de rótulo que o
YOLO de segmentação consome.

Uso:
    uv run python scripts/download_dataset.py            # dataset completo (~7k imagens)
    uv run python scripts/download_dataset.py --limit 40 # subconjunto para teste rápido
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "foodseg103"

# Polígonos com menos área que isso (em px) são ruído da máscara e são descartados.
MIN_CONTOUR_AREA = 200


def mask_to_yolo_polygons(mask: np.ndarray) -> list[str]:
    """Converte máscara binária em linhas de rótulo YOLO-seg (classe 0 + polígono normalizado)."""
    h, w = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for contour in contours:
        if cv2.contourArea(contour) < MIN_CONTOUR_AREA:
            continue
        # Simplifica o contorno para não gerar rótulos com milhares de vértices.
        epsilon = 0.002 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) < 3:
            continue
        points = approx.reshape(-1, 2).astype(np.float64)
        points[:, 0] /= w
        points[:, 1] /= h
        coords = " ".join(f"{x:.5f} {y:.5f}" for x, y in points)
        lines.append(f"0 {coords}")
    return lines


def convert_split(dataset_split, split_name: str, limit: int | None) -> None:
    images_dir = DATA_DIR / "images" / split_name
    labels_dir = DATA_DIR / "labels" / split_name
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    total = len(dataset_split) if limit is None else min(limit, len(dataset_split))
    skipped = 0
    for i in tqdm(range(total), desc=f"Convertendo {split_name}"):
        sample = dataset_split[i]
        image = sample["image"].convert("RGB")
        label = np.array(sample["label"])
        binary = (label > 0).astype(np.uint8)

        lines = mask_to_yolo_polygons(binary)
        if not lines:
            skipped += 1
            continue

        image.save(images_dir / f"{split_name}_{i:05d}.jpg", quality=90)
        (labels_dir / f"{split_name}_{i:05d}.txt").write_text("\n".join(lines))

    print(f"{split_name}: {total - skipped} imagens convertidas, {skipped} sem comida detectável")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                        help="converte só as N primeiras imagens de cada split (teste rápido)")
    args = parser.parse_args()

    print("Baixando FoodSeg103 do Hugging Face (primeira vez pode demorar)...")
    dataset = load_dataset("EduardoPacheco/FoodSeg103")

    for hf_split, yolo_split in [("train", "train"), ("validation", "val")]:
        convert_split(dataset[hf_split], yolo_split, args.limit)

    data_yaml = DATA_DIR / "data.yaml"
    data_yaml.write_text(
        f"path: {DATA_DIR.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: food\n"
    )
    print(f"Pronto. Configuração do dataset em {data_yaml}")


if __name__ == "__main__":
    main()
