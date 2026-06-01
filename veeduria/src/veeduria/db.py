import sqlite3
from pathlib import Path


def init_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS actas (
            id_transmission_code TEXT PRIMARY KEY,
            pdf_path             TEXT NOT NULL,
            depto                TEXT,
            municipio            TEXT,
            zona                 TEXT,
            puesto               TEXT,
            mesa                 TEXT,
            procesada_ocr        INTEGER DEFAULT 0,
            creada_en            TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS campos (
            id_transmission_code    TEXT PRIMARY KEY REFERENCES actas,
            candidato_1             INTEGER,
            candidato_2             INTEGER,
            candidato_3             INTEGER,
            candidato_4             INTEGER,
            candidato_5             INTEGER,
            votos_en_blanco         INTEGER,
            votos_nulos             INTEGER,
            votos_totales_depositados INTEGER,
            total_votantes_sufragaron INTEGER,
            potencial_votantes      INTEGER,
            ocr_raw                 TEXT,
            confianza               REAL,
            extraida_en             TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS flags (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            id_transmission_code TEXT REFERENCES actas,
            tipo            TEXT,
            detalle         TEXT,
            diferencia      INTEGER DEFAULT 0,
            inspeccion_visual TEXT,
            creada_en       TEXT DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_flags_code ON flags(id_transmission_code);
    """)
    conn.commit()
    return conn


def insert_acta(conn: sqlite3.Connection, *, id_transmission_code: str,
                pdf_path: str, depto: str, municipio: str,
                zona: str, puesto: str, mesa: str) -> None:
    conn.execute(
        """INSERT OR IGNORE INTO actas
           (id_transmission_code, pdf_path, depto, municipio, zona, puesto, mesa)
           VALUES (?,?,?,?,?,?,?)""",
        (id_transmission_code, pdf_path, depto, municipio, zona, puesto, mesa),
    )
    conn.commit()


def insert_campos(conn: sqlite3.Connection, *, id_transmission_code: str,
                  campos: dict, ocr_raw: str, confianza: float) -> None:
    conn.execute(
        """INSERT OR REPLACE INTO campos
           (id_transmission_code, candidato_1, candidato_2, candidato_3,
            candidato_4, candidato_5, votos_en_blanco, votos_nulos,
            votos_totales_depositados, total_votantes_sufragaron,
            potencial_votantes, ocr_raw, confianza)
           VALUES (:id,:c1,:c2,:c3,:c4,:c5,:blanco,:nulos,:total,:sufragaron,:potencial,:raw,:conf)""",
        {
            "id": id_transmission_code,
            "c1": campos.get("candidato_1", 0),
            "c2": campos.get("candidato_2", 0),
            "c3": campos.get("candidato_3", 0),
            "c4": campos.get("candidato_4", 0),
            "c5": campos.get("candidato_5", 0),
            "blanco": campos.get("votos_en_blanco", 0),
            "nulos": campos.get("votos_nulos", 0),
            "total": campos.get("votos_totales_depositados", 0),
            "sufragaron": campos.get("total_votantes_sufragaron", 0),
            "potencial": campos.get("potencial_votantes", 0),
            "raw": ocr_raw,
            "conf": confianza,
        },
    )
    conn.execute(
        "UPDATE actas SET procesada_ocr=1 WHERE id_transmission_code=?",
        (id_transmission_code,),
    )
    conn.commit()


def insert_flag(conn: sqlite3.Connection, *, id_transmission_code: str,
                tipo: str, detalle: str, diferencia: int = 0) -> None:
    conn.execute(
        "INSERT INTO flags (id_transmission_code, tipo, detalle, diferencia) VALUES (?,?,?,?)",
        (id_transmission_code, tipo, detalle, diferencia),
    )
    conn.commit()


def update_flag_inspeccion(conn: sqlite3.Connection, *,
                           id_transmission_code: str, inspeccion: str) -> None:
    conn.execute(
        "UPDATE flags SET inspeccion_visual=? WHERE id_transmission_code=?",
        (inspeccion, id_transmission_code),
    )
    conn.commit()


def get_flagged_actas(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        """SELECT a.*, f.tipo, f.detalle, f.diferencia, f.inspeccion_visual
           FROM flags f JOIN actas a USING (id_transmission_code)
           ORDER BY f.diferencia DESC"""
    ).fetchall()
    return [dict(r) for r in rows]


def get_campos(conn: sqlite3.Connection, id_transmission_code: str) -> dict | None:
    row = conn.execute(
        "SELECT * FROM campos WHERE id_transmission_code=?",
        (id_transmission_code,),
    ).fetchone()
    return dict(row) if row else None
