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
    uv run python scripts/predict.py --webcam            # demo ao vivo
    uv run python scripts/predict.py --webcam --gravar   # demo ao vivo gravando .mp4
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
        minRadius=min_dim // 3,   # um prato fotografado de propósito ocupa boa parte do quadro
        maxRadius=int(min_dim * 0.75),
    )
    if circles is None:
        return None
    # HoughCircles retorna os círculos em ordem decrescente de votos. Percorre nessa
    # ordem e aceita o primeiro plausível: centro na região central da imagem e
    # círculo quase inteiro dentro do quadro (um prato fotografado de propósito
    # dificilmente está encostado na borda da foto).
    for cx, cy, r in np.round(circles[0]).astype(int):
        center_ok = abs(cx - w / 2) < w / 4 and abs(cy - h / 2) < h / 4
        inside_ok = (cx - 0.7 * r >= -0.1 * w and cx + 0.7 * r <= 1.1 * w
                     and cy - 0.7 * r >= -0.1 * h and cy + 0.7 * r <= 1.1 * h)
        if center_ok and inside_ok:
            return int(cx), int(cy), int(r)
    return None


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


def annotate(model: YOLO, image: np.ndarray, conf: float) -> tuple[np.ndarray, float, str]:
    """Aplica o pipeline completo a um quadro e devolve (visualização, %, referência usada)."""
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
    return vis, percent, reference_label


def analyze_image(model: YOLO, image_path: Path, conf: float) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"[aviso] não consegui ler {image_path}, pulando")
        return

    vis, percent, reference_label = annotate(model, image, conf)
    side_by_side = np.hstack([image, vis])
    out_path = OUTPUT_DIR / f"{image_path.stem}_resultado.jpg"
    cv2.imwrite(str(out_path), side_by_side)
    print(f"{image_path.name}: {percent:.1f}% de comida ({reference_label}) -> {out_path}")


def run_webcam(model: YOLO, conf: float, camera: int, record: bool) -> None:
    """Demo ao vivo: mostra a webcam anotada em uma janela.

    Teclas: Q ou ESC encerra; S salva um instantâneo em outputs/.
    Com --gravar, o vídeo anotado inteiro é salvo em outputs/ como .mp4.
    """
    cap = cv2.VideoCapture(camera, cv2.CAP_DSHOW)  # CAP_DSHOW abre mais rápido no Windows
    if not cap.isOpened():
        raise SystemExit(f"Não consegui abrir a câmera {camera}. Tente --camera 1.")

    writer = None
    if record:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_path = OUTPUT_DIR / "demo_webcam.mp4"
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (width, height))
        print(f"Gravando em {video_path} ...")

    print("Demo ao vivo: Q/ESC encerra, S salva instantâneo.")
    snapshot_count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            vis, _, _ = annotate(model, frame, conf)
            cv2.imshow("food-plate-vision (Q para sair)", vis)
            if writer is not None:
                writer.write(vis)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):  # 27 = ESC
                break
            if key in (ord("s"), ord("S")):
                snapshot_count += 1
                snap_path = OUTPUT_DIR / f"webcam_{snapshot_count:02d}.jpg"
                cv2.imwrite(str(snap_path), vis)
                print(f"Instantâneo salvo em {snap_path}")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", default=None,
                        help="imagem ou pasta de imagens (omita ao usar --webcam)")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS),
                        help="pesos do modelo treinado (best.pt)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="confiança mínima das detecções do YOLO")
    parser.add_argument("--webcam", action="store_true",
                        help="demo ao vivo com a webcam em vez de fotos")
    parser.add_argument("--camera", type=int, default=0,
                        help="índice da câmera para --webcam")
    parser.add_argument("--gravar", action="store_true",
                        help="com --webcam, grava o vídeo anotado em outputs/demo_webcam.mp4")
    args = parser.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        raise SystemExit(f"Pesos não encontrados em {weights}. Rode antes: uv run python scripts/train.py")

    if args.webcam:
        OUTPUT_DIR.mkdir(exist_ok=True)
        run_webcam(YOLO(str(weights)), args.conf, args.camera, args.gravar)
        return

    if args.input is None:
        raise SystemExit("Informe uma imagem/pasta, ou use --webcam para a demo ao vivo.")

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
