from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    actas_dir: Path = Path("/Volumes/ssd niko/actas_e14_2026")
    db_path: Path = Path("veeduria.db")
    report_dir: Path = Path("reportes")

    # Tlama 124M — MLX Hub path (o fallback)
    tlama_model: str = "mlx-community/SmolLM2-135M-Instruct-4bit"

    # Moondream2 — HuggingFace path
    moondream_model: str = "vikhyatk/moondream2"
    moondream_revision: str = "2025-01-09"

    # Porcentaje de la distancia aritmética que activa el flag visual
    umbral_diferencia_votos: int = 0

    # Máximo de actas a inspeccionar visualmente en una corrida (RAM)
    max_inspeccion_visual: int = 500


settings = Settings()
