import sqlite3
import pytest
from pathlib import Path
from veeduria.db import init_db, insert_acta, insert_campos, insert_flag, get_flagged_actas


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "test.db"
    conn = init_db(path)
    yield conn
    conn.close()


def test_init_creates_tables(db):
    tables = {row[0] for row in db.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )}
    assert {"actas", "campos", "flags"} <= tables


def test_insert_and_query_acta(db):
    insert_acta(db, id_transmission_code="123", pdf_path="/tmp/test.pdf",
                depto="15", municipio="118", zona="000", puesto="00", mesa="027")
    row = db.execute("SELECT * FROM actas WHERE id_transmission_code='123'").fetchone()
    assert row is not None


def test_insert_campos(db):
    insert_acta(db, id_transmission_code="456", pdf_path="/tmp/x.pdf",
                depto="11", municipio="001", zona="000", puesto="00", mesa="001")
    insert_campos(db, id_transmission_code="456", campos={
        "candidato_1": 100, "candidato_2": 50, "votos_en_blanco": 5,
        "votos_nulos": 2, "votos_totales_depositados": 157,
        "total_votantes_sufragaron": 157, "potencial_votantes": 300,
    }, ocr_raw="raw text", confianza=0.92)
    row = db.execute("SELECT votos_totales_depositados FROM campos WHERE id_transmission_code='456'").fetchone()
    assert row[0] == 157


def test_insert_flag(db):
    insert_acta(db, id_transmission_code="789", pdf_path="/tmp/y.pdf",
                depto="11", municipio="001", zona="000", puesto="00", mesa="002")
    insert_flag(db, id_transmission_code="789", tipo="suma_incorrecta",
                detalle="esperado 157, encontrado 160", diferencia=3)
    flagged = get_flagged_actas(db)
    assert any(r["id_transmission_code"] == "789" for r in flagged)
