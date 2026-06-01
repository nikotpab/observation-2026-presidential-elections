from dataclasses import dataclass
from PIL import Image

# Surya 0.17.x uses Predictor objects instead of separate model/processor pairs.
# Models are loaded lazily on first use; instantiated once per process.
_det_predictor = None
_rec_predictor = None


def _load_predictors():
    global _det_predictor, _rec_predictor
    if _det_predictor is None:
        from surya.detection import DetectionPredictor
        from surya.recognition import RecognitionPredictor
        from surya.foundation import FoundationPredictor
        from surya.settings import settings

        _det_predictor = DetectionPredictor()
        _rec_predictor = RecognitionPredictor(
            FoundationPredictor(checkpoint=settings.RECOGNITION_MODEL_CHECKPOINT)
        )
    return _det_predictor, _rec_predictor


@dataclass
class LineOCR:
    text: str
    bbox: tuple[float, float, float, float]  # x0, y0, x1, y1
    confidence: float


def extract_lines(image: Image.Image, langs: list[str] = ["es"]) -> list[LineOCR]:
    """Runs Surya OCR on an image and returns recognized text lines with bboxes.

    langs is accepted for API compatibility but ignored by Surya 0.17.x,
    which handles language detection internally via its foundation model.
    """
    det_predictor, rec_predictor = _load_predictors()

    results = rec_predictor(
        [image],
        det_predictor=det_predictor,
        sort_lines=True,
    )

    lines = []
    for line in results[0].text_lines:
        bbox = line.bbox  # [x0, y0, x1, y1] from PolygonBox property
        lines.append(LineOCR(
            text=line.text,
            bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
            confidence=line.confidence,
        ))
    return lines


def full_text(image: Image.Image) -> tuple[str, float]:
    """Returns (concatenated text, mean confidence) for quick structuring."""
    lines = extract_lines(image)
    if not lines:
        return "", 0.0
    text = "\n".join(l.text for l in lines)
    conf = sum(l.confidence for l in lines) / len(lines)
    return text, conf
