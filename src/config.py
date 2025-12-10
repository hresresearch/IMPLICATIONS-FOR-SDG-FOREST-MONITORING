"""Configuration helpers for API clients.

Spanish / Español:
Este módulo centraliza las variables de configuración necesarias para los
clientes de API (por ejemplo, MapBiomas Alerta, CEPALSTAT y Global Forest
Watch). Proporciona funciones auxiliares para leer variables de entorno,
aplicar valores por defecto y generar errores claros cuando faltan
credenciales obligatorias.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env when the module is imported.
load_dotenv()


def _get_env_var(name: str, default: Optional[str] = None, required: bool = False) -> str:
    """
    Fetch an environment variable with optional default and required flag.
    Raises RuntimeError if a required variable is missing.
    """
    value = os.getenv(name, default)
    if required and not value:
        message = f"{name} is not set"
        logger.error(message)
        raise RuntimeError(message)
    return value or ""


def get_mapbiomas_alerta_token() -> str:
    """Return the MapBiomas Alerta API token."""
    return _get_env_var("MAPBIOMAS_ALERTA_TOKEN", required=True)


def get_mapbiomas_alerta_token_optional() -> Optional[str]:
    """Return the MapBiomas Alerta token if set, otherwise None."""
    token = os.getenv("MAPBIOMAS_ALERTA_TOKEN")
    return token if token else None


def get_mapbiomas_alerta_email() -> str:
    """Return the MapBiomas Alerta account email for sign-in."""
    return _get_env_var("MAPBIOMAS_ALERTA_EMAIL", required=True)


def get_mapbiomas_alerta_password() -> str:
    """Return the MapBiomas Alerta account password for sign-in."""
    return _get_env_var("MAPBIOMAS_ALERTA_PASSWORD", required=True)


def get_cepalstat_base_url() -> str:
    """Return the base URL for the CEPALSTAT API."""
    return _get_env_var(
        "CEPALSTAT_API_BASE_URL",
        default="https://api-cepalstat.cepal.org/cepalstat/api/v1",
        required=False,
    )


def get_gfw_base_url() -> str:
    """Return the base URL for the Global Forest Watch API."""
    return _get_env_var(
        "GFW_API_BASE_URL",
        default="https://data-api.globalforestwatch.org",
        required=False,
    )


def get_gfw_api_token() -> str:
    """Return the Global Forest Watch API token."""
    return _get_env_var("GFW_API_TOKEN", required=True)
