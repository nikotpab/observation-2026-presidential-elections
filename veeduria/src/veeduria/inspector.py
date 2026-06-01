from __future__ import annotations
from dataclasses import dataclass
from PIL import Image

_model = None
_tokenizer = None

PROMPT = (
    "This is a Colombian electoral tally sheet (Acta E-14). "
    "Examine it carefully for signs of physical tampering: "
    "(1) numbers crossed out and replaced, "
    "(2) white correction fluid over any entry, "
    "(3) numbers written in a different ink color or handwriting style than surrounding text, "
    "(4) erasure marks or smudging over numerical fields, "
    "(5) numbers superimposed on previous entries. "
    "Reply with exactly one word on the first line: SOSPECHOSA or LIMPIA. "
    "Then on a new line describe any specific observation, or 'Sin observaciones'."
)


def _load_moondream(model_id: str, revision: str):
    global _model, _tokenizer
    if _model is None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        _model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            revision=revision,
        )
        # Move model to MPS when available so vision_projection tensors stay
        # on the same device (vision.py adaptive_avg_pool2d moves through MPS).
        if torch.backends.mps.is_available():
            _model = _model.to("mps")
    return _model, _tokenizer


@dataclass
class InspeccionVisual:
    veredicto: str           # "SOSPECHOSA" | "LIMPIA" | "ERROR"
    observacion: str
    modelo: str


def inspeccionar(
    image: Image.Image,
    model_id: str = "vikhyatk/moondream2",
    revision: str = "2025-01-09",
) -> InspeccionVisual:
    try:
        model, tokenizer = _load_moondream(model_id, revision)
        enc = model.encode_image(image)
        respuesta = model.answer_question(enc, PROMPT, tokenizer)

        lines = respuesta.strip().splitlines()
        veredicto = lines[0].strip().upper() if lines else "ERROR"
        if veredicto not in ("SOSPECHOSA", "LIMPIA"):
            veredicto = "ERROR"
        observacion = lines[1].strip() if len(lines) > 1 else respuesta.strip()

        return InspeccionVisual(
            veredicto=veredicto,
            observacion=observacion,
            modelo=model_id,
        )
    except Exception as exc:
        return InspeccionVisual(
            veredicto="ERROR",
            observacion=str(exc),
            modelo=model_id,
        )
