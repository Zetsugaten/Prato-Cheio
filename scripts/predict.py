"""Estima a quantidade de comida no prato de uma foto.

Pipeline:
  1. O YOLO-seg fine-tunado segmenta a comida (máscara binária).
  2. A borda do prato é detectada pela transformada de Hough circular
     (processamento clássico de sinais: gradiente + votação no espaço de parâmetros).
  3. Quantidade = pixels de comida dentro do prato / área do prato.

Se nenhum círculo for encontrado (prato fora de quadro, formato não circular),
o script usa a imagem inteira como referência e avisa no resultado.

Uso:
    uv run python scripts/predict.py foto.jpg
    uv run python scripts/predict.py pasta_com_fotos/ --weights runs/food-seg/weights/best.pt
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = ROOT / "runs" / "food-seg" / "weights" / "best.pt"
OUTPUT_DIR = ROOT / "outputs"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def detect_plate(image: np.ndarray) -> tuple[int, int, int] | None:
    """Detecta o maior círculo (borda do prato) via transformada de Hough.

    Retorna (cx, cy, raio) em pixels, ou None se nenhum círculo plausível existir.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7)
    h, w = gray.shape
    min_dim = min(h, w)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.5,
        minDist=min_dim // 2,
        param1=120,   # limiar do detector de bordas (Canny) interno
        param2=60,    # limiar de votos no acumulador de Hough
        minRadius=min_dim // 5,
        maxRadius=int(min_dim * 0.7),
    )
    if circles is None:
        return None
    # O prato tende a ser o maior círculo com muitos votos; pega o de maior raio.
    cx, cy, r = max(np.round(circles[0]).astype(int), key=lambda c: c[2])
    return int(cx), int(cy), int(r)


def food_mask_from_yolo(model: YOLO, image: np.ndarray, conf: float) -> np.ndarray:
    """Roda o YOLO-seg e devolve a união das máscaras de comida (uint8, 0 ou 1)."""
    h, w = image.shape[:2]
    result = model.predict(image, conf=conf, verbose=False)[0]
    mask = np.zeros((h, w), dtype=np.uint8)
    if result.masks is not None:
        for m in result.masks.data.cpu().numpy():
            resized = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            mask |= resized.astype(np.uint8)
    return mask


def analyze_image(model: YOLO, image_path: Path, conf: float) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"[aviso] não consegui ler {image_path}, pulando")
        return
    h, w = image.shape[:2]

    food_mask = food_mask_from_yolo(model, image, conf)
    plate = detect_plate(image)

    if plate is not None:
        cx, cy, r = plate
        plate_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(plate_mask, (cx, cy), r, 1, thickness=-1)
        reference_label = "prato (Hough)"
    else:
        plate_mask = np.ones((h, w), dtype=np.uint8)
        reference_label = "imagem inteira (prato nao detectado)"

    plate_area = int(plate_mask.sum())
    food_area = int((food_mask & plate_mask).sum())
    percent = 100.0 * food_area / plate_area if plate_area else 0.0

    # Visualização: comida em verde translúcido, borda do prato em azul, % no topo.
    overlay = image.copy()
    overlay[(food_mask & plate_mask) == 1] = (0, 200, 0)
    vis = cv2.addWeighted(overlay, 0.45, image, 0.55, 0)
    if plate is not None:
        cv2.circle(vis, (cx, cy), r, (255, 120, 0), 3)
    banner = f"comida: {percent:.1f}% do {reference_label}"
    cv2.rectangle(vis, (0, 0), (w, 42), (0, 0, 0), -1)
    cv2.putText(vis, banner, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    side_by_side = np.hstack([image, vis])
    out_path = OUTPUT_DIR / f"{image_path.stem}_resultado.jpg"
    cv2.imwrite(str(out_path), side_by_side)
    print(f"{image_path.name}: {percent:.1f}% de comida ({reference_label}) -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="imagem ou pasta de imagens")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS),
                        help="pesos do modelo treinado (best.pt)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="confiança mínima das detecções do YOLO")
    args = parser.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        raise SystemExit(f"Pesos não encontrados em {weights}. Rode antes: uv run python scripts/train.py")

    input_path = Path(args.input)
    if input_path.is_dir():
        images = sorted(p for p in input_path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    else:
        images = [input_path]
    if not images:
        raise SystemExit(f"Nenhuma imagem encontrada em {input_path}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    model = YOLO(str(weights))
    for image_path in images:
        analyze_image(model, image_path, args.conf)


if __name__ == "__main__":
    main()
