import json
import pytest
from pathlib import Path
from veeduria.reporter import generar_reporte, ResumenReporte


@pytest.fixture
def hallazgos_muestra():
    return [
        {
            "id_transmission_code": "123",
            "depto": "15", "municipio": "118",
            "zona": "000", "puesto": "00", "mesa": "027",
            "pdf_path": "/Volumes/ssd niko/actas_e14_2026/E14_PRE_15_118_000_00_027.pdf",
            "tipo": "suma_incorrecta",
            "detalle": "Suma de votos (1777) != votos_totales_depositados (1800). Diferencia: 23",
            "diferencia": 23,
            "inspeccion_visual": "SOSPECHOSA — Se observa número sobrescrito en campo VOTOS TOTALES",
        },
    ]


def test_generar_reporte_crea_html(tmp_path, hallazgos_muestra):
    reporte = generar_reporte(hallazgos_muestra, output_dir=tmp_path)
    assert reporte.html_path.exists()
    content = reporte.html_path.read_text()
    assert "suma_incorrecta" in content
    assert "15" in content  # depto


def test_generar_reporte_crea_json(tmp_path, hallazgos_muestra):
    reporte = generar_reporte(hallazgos_muestra, output_dir=tmp_path)
    assert reporte.json_path.exists()
    data = json.loads(reporte.json_path.read_text())
    assert data["total_flaggeadas"] == 1


def test_reporte_vacio(tmp_path):
    reporte = generar_reporte([], output_dir=tmp_path)
    assert reporte.html_path.exists()
    data = json.loads(reporte.json_path.read_text())
    assert data["total_flaggeadas"] == 0
