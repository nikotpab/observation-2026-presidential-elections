#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "httpx[http2]",
#   "tqdm",
#   "python-dotenv",
# ]
# ///
"""
Descarga masiva de actas E-14 - Elecciones Presidenciales Colombia 2026.

Construye las URLs a partir del indice de transmision local y descarga cada
acta como PDF usando el nombre legible E14_PRE_<depto>_<mun>_<zona>_<puesto>_<mesa>.pdf,
formato conveniente para pipelines de vision artificial.

Uso:
    uv run tally-sheets-download.py [OPCIONES]

Opciones:
    --index FILE        Archivo de indice JSON  (default: INDEX_FILE del .env o e14_transmission_index_2026.json)
    --out DIR           Directorio de salida    (default: OUTPUT_DIR del .env o ./actas)
    --concurrency N     Descargas paralelas     (default: CONCURRENCY del .env o 20)
    --retries N         Reintentos por archivo  (default: RETRIES del .env o 3)
    --limit N           Limitar a N registros   (util para pruebas)
    --status N          Filtrar por idTransmissionCodeStatus (ej: 11)
    --errors FILE       Ruta del log de errores (default: ERRORS_FILE del .env o errors.csv)
    --env FILE          Archivo .env alternativo (default: .env)
"""

import argparse
import asyncio
import csv
import json
from pathlib import Path
from typing import Optional

import httpx
from dotenv import dotenv_values
from tqdm import tqdm

BASE_URL = "https://divulgacione14presidente.registraduria.gov.co/assets/temis/pdf"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "application/pdf,*/*",
    "Referer": "https://divulgacione14presidente.registraduria.gov.co/",
}


def build_url(record: dict) -> str:
    depto = record["idDepartmentCode"]
    mun = record["municipalityCode"]
    zona = record["idZoneCode"].zfill(3)
    puesto = record["standCode"].zfill(2)
    mesa = record["numberStand"].zfill(3)
    nombre_pdf = record["expectedName"]
    return f"{BASE_URL}/{depto}/{mun}/{zona}/{puesto}/{mesa}/PRE/{nombre_pdf}"


def local_path(out_dir: Path, record: dict) -> Path:
    depto = record["idDepartmentCode"]
    mun = record["municipalityCode"]
    zona = record["idZoneCode"].zfill(3)
    puesto = record["standCode"].zfill(2)
    mesa = record["numberStand"].zfill(3)
    nombre = f"E14_PRE_{depto}_{mun}_{zona}_{puesto}_{mesa}.pdf"
    return out_dir / nombre


def load_records(index_path: str, status_filter: Optional[int]) -> list:
    with open(index_path, encoding="utf-8") as f:
        raw = json.load(f)

    data = raw.get("data", raw)
    records = []

    if isinstance(data, dict):
        for bucket in data.values():
            nodes = bucket.get("nodes", bucket) if isinstance(bucket, dict) else bucket
            records.extend(nodes)
    elif isinstance(data, list):
        records = data

    if status_filter is not None:
        records = [r for r in records if r.get("idTransmissionCodeStatus") == status_filter]

    return records


async def download_one(
    client: httpx.AsyncClient,
    record: dict,
    dest: Path,
    retries: int,
    semaphore: asyncio.Semaphore,
) -> tuple:
    """Descarga un PDF. Retorna (exito, mensaje_error)."""
    if dest.exists() and dest.stat().st_size > 0:
        return True, None

    url = build_url(record)
    dest.parent.mkdir(parents=True, exist_ok=True)

    async with semaphore:
        for attempt in range(1, retries + 1):
            try:
                async with client.stream("GET", url, timeout=20.0) as response:
                    if response.status_code == 404:
                        return False, f"404 {url}"
                    response.raise_for_status()
                    tmp = dest.with_suffix(".tmp")
                    with open(tmp, "wb") as fh:
                        async for chunk in response.aiter_bytes(chunk_size=65536):
                            fh.write(chunk)
                    tmp.rename(dest)
                    return True, None
            except (httpx.HTTPStatusError, httpx.RequestError) as exc:
                if attempt == retries:
                    return False, f"{type(exc).__name__}: {exc}"
                await asyncio.sleep(2 ** attempt)

    return False, "max reintentos agotados"


async def run(args: argparse.Namespace) -> None:
    records = load_records(args.index, args.status)
    print(f"Registros cargados del indice: {len(records):,}")

    if args.limit:
        records = records[: args.limit]
        print(f"Limitado a {args.limit} registros (--limit).")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    semaphore = asyncio.Semaphore(args.concurrency)
    errors = []
    counters = {"ok": 0, "skip": 0, "error": 0}

    limits = httpx.Limits(
        max_connections=args.concurrency + 10,
        max_keepalive_connections=args.concurrency,
    )
    async with httpx.AsyncClient(headers=HEADERS, limits=limits, follow_redirects=True) as client:

        async def tracked(record):
            dest = local_path(out_dir, record)
            if dest.exists() and dest.stat().st_size > 0:
                counters["skip"] += 1
                return
            success, err = await download_one(client, record, dest, args.retries, semaphore)
            if success:
                counters["ok"] += 1
            else:
                counters["error"] += 1
                errors.append({
                    "idTransmissionCode": record.get("idTransmissionCode"),
                    "url": build_url(record),
                    "error": err,
                })

        with tqdm(total=len(records), unit="acta", dynamic_ncols=True) as bar:
            async def tracked_with_bar(record):
                await tracked(record)
                bar.update(1)
                bar.set_postfix(**counters)

            await asyncio.gather(*[tracked_with_bar(r) for r in records])

    print(
        f"\nFinalizado — descargadas: {counters['ok']:,}  "
        f"omitidas (ya existian): {counters['skip']:,}  "
        f"errores: {counters['error']:,}"
    )

    if errors:
        with open(args.errors, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["idTransmissionCode", "url", "error"])
            writer.writeheader()
            writer.writerows(errors)
        print(f"Log de errores: {args.errors}")


def main() -> None:
    env = dotenv_values(".env")

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--index",
        default=env.get("INDEX_FILE", "e14_transmission_index_2026.json"),
    )
    parser.add_argument(
        "--out",
        default=env.get("OUTPUT_DIR", "./actas"),
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=int(env.get("CONCURRENCY", 20)),
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=int(env.get("RETRIES", 3)),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--status",
        type=int,
        default=int(env["STATUS_FILTER"]) if "STATUS_FILTER" in env else None,
        help="Filtrar por idTransmissionCodeStatus (ej: 11 = transmitidas y confirmadas)",
    )
    parser.add_argument(
        "--errors",
        default=env.get("ERRORS_FILE", "errors.csv"),
    )
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
