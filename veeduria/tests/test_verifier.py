import pytest
from veeduria.structurer import ActaCampos
from veeduria.verifier import verificar, ViolacionAritmetica


def _make_campos(**kwargs) -> ActaCampos:
    defaults = dict(
        votos_por_candidato={"A": 1000, "B": 500, "C": 200},
        votos_en_blanco=50,
        votos_nulos=27,
        votos_totales_depositados=1777,
        total_votantes_sufragaron=1777,
        potencial_votantes=3000,
        confianza=0.92,
    )
    defaults.update(kwargs)
    return ActaCampos(**defaults)


def test_acta_limpia_sin_violaciones():
    assert verificar(_make_campos()) == []


def test_detecta_suma_incorrecta():
    campos = _make_campos(votos_totales_depositados=1800)
    violaciones = verificar(campos)
    assert any(v.tipo == "suma_incorrecta" for v in violaciones)
    assert any(v.diferencia == 23 for v in violaciones)


def test_detecta_sufragantes_distinto_a_total():
    campos = _make_campos(total_votantes_sufragaron=1800)
    violaciones = verificar(campos)
    assert any(v.tipo == "sufragantes_distinto_total" for v in violaciones)


def test_detecta_sufragantes_mayor_potencial():
    campos = _make_campos(total_votantes_sufragaron=3100, potencial_votantes=3000)
    violaciones = verificar(campos)
    assert any(v.tipo == "sufragantes_excede_potencial" for v in violaciones)


def test_violacion_tiene_descripcion_legible():
    campos = _make_campos(votos_totales_depositados=1800)
    v = verificar(campos)[0]
    assert str(v.diferencia) in v.detalle
