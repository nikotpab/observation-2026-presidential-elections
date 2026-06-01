from __future__ import annotations
import re
from dataclasses import dataclass, field


# Threshold below which we mark the acta for manual review
CONFIANZA_MINIMA = 0.65

# Regex patterns for known E-14 field labels (case-insensitive, accent-tolerant)
_PATRON_NUMERO = r"(\d[\d\s]{0,4}\d|\d)"  # handles "1 247" formatting artifacts

_PATRONES = {
    "votos_en_blanco":            re.compile(r"votos?\s+en\s+blanco\D{0,30}" + _PATRON_NUMERO, re.I),
    "votos_nulos":                re.compile(r"votos?\s+nul[oa]s?\D{0,30}" + _PATRON_NUMERO, re.I),
    "votos_totales_depositados":  re.compile(r"totales?\s+depositados?\D{0,30}" + _PATRON_NUMERO, re.I),
    "total_votantes_sufragaron":  re.compile(r"sufrag[a-z]+\D{0,30}" + _PATRON_NUMERO, re.I),
    "potencial_votantes":         re.compile(r"potencial\s+de\s+votantes\D{0,30}" + _PATRON_NUMERO, re.I),
}

# Candidate rows: any line that has text + a number and doesn't match special rows
_PATRON_CANDIDATO = re.compile(
    r"^(?!.*(?:blanco|nulo|total|potencial|sufrag|acta|jurado|fecha|hora|firma))(.{5,40}?)\s+(\d[\d\s]{0,5}\d|\d{1,6})\s*$",
    re.I | re.MULTILINE,
)


def _limpiar_numero(s: str) -> int:
    return int(re.sub(r"\s", "", s))


@dataclass
class ActaCampos:
    votos_por_candidato: dict[str, int] = field(default_factory=dict)
    votos_en_blanco: int = 0
    votos_nulos: int = 0
    votos_totales_depositados: int = 0
    total_votantes_sufragaron: int = 0
    potencial_votantes: int = 0
    ocr_raw: str = ""
    confianza: float = 0.0
    requiere_revision_manual: bool = False


def parse_campos(texto: str, confianza: float) -> ActaCampos | None:
    if not texto or not texto.strip():
        return None

    campos = ActaCampos(ocr_raw=texto, confianza=confianza)
    campos.requiere_revision_manual = confianza < CONFIANZA_MINIMA

    # Extract fixed special fields
    for campo, patron in _PATRONES.items():
        m = patron.search(texto)
        if m:
            setattr(campos, campo, _limpiar_numero(m.group(1)))

    # Extract candidate rows heuristically
    for m in _PATRON_CANDIDATO.finditer(texto):
        nombre = m.group(1).strip()
        votos = _limpiar_numero(m.group(2))
        if votos > 0:
            campos.votos_por_candidato[nombre] = votos

    # If both candidate votes and blanco/nulos are zero, mark for review
    if not campos.votos_por_candidato and campos.votos_totales_depositados == 0:
        campos.requiere_revision_manual = True

    return campos
