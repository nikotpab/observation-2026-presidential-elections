from __future__ import annotations
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer
from tqdm import tqdm

from veeduria.config import settings
from veeduria.db import (
    init_db, insert_acta, insert_campos, insert_flag,
    update_flag_inspeccion, get_flagged_actas,
)

app = typer.Typer(help="Veeduria Electoral — pipeline de deteccion de fraude en actas E-14")


def _get_db():
    return init_db(settings.db_path)


@app.command()
def ocr(
    actas_dir: Annotated[Optional[Path], typer.Option(help="Sobreescribe ACTAS_DIR del .env")] = None,
    limit: Annotated[Optional[int], typer.Option(help="Procesar solo N actas (pruebas)")] = None,
):
    """Etapa 1: convierte PDFs a texto via OCR y almacena campos en la BD."""
    from veeduria.pdf_to_images import page_one
    from veeduria.ocr import full_text
    from veeduria.structurer import parse_campos

    directorio = actas_dir or settings.actas_dir
    pdfs = sorted(directorio.glob("E14_PRE_*.pdf"))
    if limit:
        pdfs = pdfs[:limit]

    typer.echo(f"Procesando {len(pdfs):,} PDFs desde {directorio}")
    conn = _get_db()
    ok = err = skip = 0

    for pdf in tqdm(pdfs, unit="acta"):
        partes = pdf.stem.split("_")  # E14_PRE_depto_mun_zona_puesto_mesa
        if len(partes) != 7:
            err += 1
            continue

        _, _, depto, mun, zona, puesto, mesa = partes
        code = f"{depto}_{mun}_{zona}_{puesto}_{mesa}"

        row = conn.execute(
            "SELECT procesada_ocr FROM actas WHERE id_transmission_code=?", (code,)
        ).fetchone()
        if row and row[0] == 1:
            skip += 1
            continue

        insert_acta(conn, id_transmission_code=code, pdf_path=str(pdf),
                    depto=depto, municipio=mun, zona=zona, puesto=puesto, mesa=mesa)
        try:
            img = page_one(pdf)
            texto, conf = full_text(img)
            campos = parse_campos(texto, conf)
            if campos:
                insert_campos(conn, id_transmission_code=code,
                              campos={
                                  **{f"candidato_{i+1}": v
                                     for i, v in enumerate(campos.votos_por_candidato.values())},
                                  "votos_en_blanco": campos.votos_en_blanco,
                                  "votos_nulos": campos.votos_nulos,
                                  "votos_totales_depositados": campos.votos_totales_depositados,
                                  "total_votantes_sufragaron": campos.total_votantes_sufragaron,
                                  "potencial_votantes": campos.potencial_votantes,
                              },
                              ocr_raw=campos.ocr_raw,
                              confianza=campos.confianza)
            ok += 1
        except Exception as e:
            typer.echo(f"  ERROR {pdf.name}: {e}", err=True)
            err += 1

    typer.echo(f"\nOCR completo — ok: {ok:,}  omitidas: {skip:,}  errores: {err:,}")


@app.command()
def verify():
    """Etapa 2: verifica invariantes aritmeticos y registra flags en la BD."""
    from veeduria.structurer import ActaCampos
    from veeduria.verifier import verificar

    conn = _get_db()
    filas = conn.execute(
        "SELECT c.*, a.depto FROM campos c JOIN actas a USING (id_transmission_code)"
    ).fetchall()

    typer.echo(f"Verificando {len(filas):,} actas procesadas...")
    flaggeadas = 0

    for fila in tqdm(filas, unit="acta"):
        d = dict(fila)
        campos = ActaCampos(
            votos_por_candidato={
                f"candidato_{i}": d[f"candidato_{i}"] or 0
                for i in range(1, 6) if d.get(f"candidato_{i}")
            },
            votos_en_blanco=d["votos_en_blanco"] or 0,
            votos_nulos=d["votos_nulos"] or 0,
            votos_totales_depositados=d["votos_totales_depositados"] or 0,
            total_votantes_sufragaron=d["total_votantes_sufragaron"] or 0,
            potencial_votantes=d["potencial_votantes"] or 0,
            confianza=d["confianza"] or 0.0,
        )
        for v in verificar(campos):
            insert_flag(conn, id_transmission_code=d["id_transmission_code"],
                        tipo=v.tipo, detalle=v.detalle, diferencia=v.diferencia)
            flaggeadas += 1

    typer.echo(f"Verificacion completa — actas flaggeadas: {flaggeadas:,}")


@app.command()
def inspect(
    max_actas: Annotated[Optional[int], typer.Option(help="Limitar inspeccion visual a N actas")] = None,
):
    """Etapa 3: inspecciona visualmente las actas flaggeadas con Moondream2."""
    from veeduria.pdf_to_images import page_one
    from veeduria.inspector import inspeccionar

    conn = _get_db()
    flaggeadas = get_flagged_actas(conn)
    limite = max_actas or settings.max_inspeccion_visual
    a_inspeccionar = [f for f in flaggeadas if not f.get("inspeccion_visual")][:limite]

    typer.echo(f"Inspeccionando {len(a_inspeccionar)} actas flaggeadas (limite={limite})...")
    typer.echo("Nota: primera ejecucion descarga ~3.5 GB del modelo Moondream2.")

    for hallazgo in tqdm(a_inspeccionar, unit="acta"):
        pdf = Path(hallazgo["pdf_path"])
        if not pdf.exists():
            continue
        try:
            img = page_one(pdf)
            resultado = inspeccionar(img, settings.moondream_model, settings.moondream_revision)
            texto = f"{resultado.veredicto} — {resultado.observacion}"
            update_flag_inspeccion(conn, id_transmission_code=hallazgo["id_transmission_code"],
                                   inspeccion=texto)
        except Exception as e:
            update_flag_inspeccion(conn, id_transmission_code=hallazgo["id_transmission_code"],
                                   inspeccion=f"ERROR: {e}")

    typer.echo("Inspeccion visual completa.")


@app.command()
def report(
    output_dir: Annotated[Optional[Path], typer.Option()] = None,
):
    """Etapa 4: genera reporte HTML y JSON con todos los hallazgos."""
    from veeduria.reporter import generar_reporte

    conn = _get_db()
    hallazgos = get_flagged_actas(conn)
    total_procesadas = conn.execute("SELECT COUNT(*) FROM actas WHERE procesada_ocr=1").fetchone()[0]
    directorio = output_dir or settings.report_dir

    reporte = generar_reporte(hallazgos, output_dir=directorio, total_procesadas=total_procesadas)
    typer.echo(f"Reporte generado: {reporte.html_path}")
    typer.echo(f"JSON: {reporte.json_path}")
    typer.echo(f"Total flaggeadas: {reporte.total_flaggeadas}")


@app.command(name="run-all")
def run_all(
    actas_dir: Annotated[Optional[Path], typer.Option()] = None,
    limit: Annotated[Optional[int], typer.Option()] = None,
    skip_inspect: Annotated[bool, typer.Option("--skip-inspect")] = False,
):
    """Ejecuta el pipeline completo: ocr -> verify -> inspect -> report."""
    ocr(actas_dir=actas_dir, limit=limit)
    verify()
    if not skip_inspect:
        inspect()
    report()


if __name__ == "__main__":
    app()
