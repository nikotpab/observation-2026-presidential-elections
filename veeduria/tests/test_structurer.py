import pytest
from veeduria.structurer import parse_campos, ActaCampos


TEXTO_LIMPIO = """
ACTA DE ESCRUTINIO DE LOS JURADOS DE VOTACION
ELECCION PRESIDENCIA Y VICEPRESIDENCIA DE LA REPUBLICA
CANDIDATO UNO               1247
CANDIDATO DOS               892
CANDIDATO TRES              315
CANDIDATO CUATRO            88
CANDIDATO CINCO             42
VOTOS EN BLANCO             156
VOTOS NULOS                 37
VOTOS TOTALES DEPOSITADOS   2777
TOTAL VOTANTES QUE SUFRAGARON 2777
POTENCIAL DE VOTANTES       3500
"""

TEXTO_SUMA_INCORRECTA = TEXTO_LIMPIO.replace("2777", "2800")


def test_parse_campos_extrae_totales():
    campos = parse_campos(TEXTO_LIMPIO, confianza=0.95)
    assert campos.votos_totales_depositados == 2777
    assert campos.potencial_votantes == 3500
    assert campos.votos_en_blanco == 156
    assert campos.votos_nulos == 37


def test_parse_campos_extrae_candidatos():
    campos = parse_campos(TEXTO_LIMPIO, confianza=0.95)
    assert sum(campos.votos_por_candidato.values()) == 2584  # 1247+892+315+88+42


def test_parse_campos_confianza_baja_marca_revision():
    campos = parse_campos(TEXTO_LIMPIO, confianza=0.45)
    assert campos.requiere_revision_manual is True


def test_parse_campos_texto_vacio_retorna_none():
    assert parse_campos("", confianza=0.0) is None
