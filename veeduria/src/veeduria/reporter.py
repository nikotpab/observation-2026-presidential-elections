from __future__ import annotations
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader


_TEMPLATE_DIR = Path(__file__).parent / "templates"


@dataclass
class ResumenReporte:
    html_path: Path
    json_path: Path
    total_flaggeadas: int


def generar_reporte(
    hallazgos: list[dict],
    output_dir: Path,
    total_procesadas: int = 0,
) -> ResumenReporte:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    generado_en = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    env = Environment(loader=FileSystemLoader(str(_TEMPLATE_DIR)))
    template = env.get_template("report.html.j2")
    html = template.render(
        hallazgos=hallazgos,
        total_flaggeadas=len(hallazgos),
        total_procesadas=total_procesadas,
        generado_en=generado_en,
    )

    html_path = output_dir / f"reporte_{timestamp}.html"
    json_path = output_dir / f"reporte_{timestamp}.json"

    html_path.write_text(html, encoding="utf-8")
    json_path.write_text(
        json.dumps({
            "generado_en": generado_en,
            "total_flaggeadas": len(hallazgos),
            "total_procesadas": total_procesadas,
            "hallazgos": hallazgos,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return ResumenReporte(html_path=html_path, json_path=json_path,
                          total_flaggeadas=len(hallazgos))
