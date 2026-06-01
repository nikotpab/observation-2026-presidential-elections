from __future__ import annotations
from dataclasses import dataclass
from veeduria.structurer import ActaCampos


@dataclass
class ViolacionAritmetica:
    tipo: str
    detalle: str
    diferencia: int


def verificar(campos: ActaCampos) -> list[ViolacionAritmetica]:
    """
    Checks three arithmetic invariants of an E-14 tally sheet.
    Returns an empty list for a clean acta.
    """
    violaciones: list[ViolacionAritmetica] = []

    suma_votos = (
        sum(campos.votos_por_candidato.values())
        + campos.votos_en_blanco
        + campos.votos_nulos
    )

    if suma_votos != campos.votos_totales_depositados:
        diff = abs(suma_votos - campos.votos_totales_depositados)
        violaciones.append(ViolacionAritmetica(
            tipo="suma_incorrecta",
            detalle=(
                f"Suma de votos ({suma_votos}) != "
                f"votos_totales_depositados ({campos.votos_totales_depositados}). "
                f"Diferencia: {diff}"
            ),
            diferencia=diff,
        ))

    if campos.votos_totales_depositados != campos.total_votantes_sufragaron:
        diff = abs(campos.votos_totales_depositados - campos.total_votantes_sufragaron)
        violaciones.append(ViolacionAritmetica(
            tipo="sufragantes_distinto_total",
            detalle=(
                f"votos_totales_depositados ({campos.votos_totales_depositados}) != "
                f"total_votantes_sufragaron ({campos.total_votantes_sufragaron}). "
                f"Diferencia: {diff}"
            ),
            diferencia=diff,
        ))

    if (campos.potencial_votantes > 0
            and campos.total_votantes_sufragaron > campos.potencial_votantes):
        diff = campos.total_votantes_sufragaron - campos.potencial_votantes
        violaciones.append(ViolacionAritmetica(
            tipo="sufragantes_excede_potencial",
            detalle=(
                f"total_votantes_sufragaron ({campos.total_votantes_sufragaron}) > "
                f"potencial_votantes ({campos.potencial_votantes}). "
                f"Exceso: {diff}"
            ),
            diferencia=diff,
        ))

    return violaciones
