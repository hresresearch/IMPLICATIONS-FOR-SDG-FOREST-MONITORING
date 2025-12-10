"""Recursive data ingestor for ZIP archives and target geospatial/tabular files.

This module traverses a root directory, detects ZIP archives, extracts them
recursively into temporary directories, and yields paths to target files
(.csv, .shp, .tif, .geojson, .dbf, .shx). It standardizes to UTF-8 paths and
optionally computes SHA-256 checksums for integrity logging.

Spanish / Español:
Este módulo recorre de forma recursiva un directorio raíz, detecta archivos
ZIP, los extrae en directorios temporales y devuelve las rutas de los
archivos objetivo (por ejemplo, CSV, shapefiles, GeoTIFF, GeoJSON). Además
puede calcular sumas de comprobación SHA‑256 para documentar la integridad
de los datos y facilitar la reproducción del paquete.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Generator, Iterable, Optional, Set


TARGET_EXTENSIONS: Set[str] = {".csv", ".shp", ".tif", ".geojson", ".dbf", ".shx"}


def sha256sum(path: Path, chunk_size: int = 8192) -> str:
    """Compute SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _extract_zip(zip_path: Path, temp_dir: Path) -> Path:
    """Extract a ZIP file into a subdirectory of temp_dir and return the extraction path."""
    dest = temp_dir / zip_path.stem
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)
    return dest


def _iter_files_recursive(root: Path) -> Generator[Path, None, None]:
    """Yield all files under root (non-zip) and nested zip contents."""
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp_dir = Path(tmp_str)
        stack = [root]
        while stack:
            current = stack.pop()
            if current.is_file():
                if current.suffix.lower() == ".zip":
                    extracted = _extract_zip(current, tmp_dir)
                    stack.append(extracted)
                else:
                    yield current
            elif current.is_dir():
                for child in current.iterdir():
                    stack.append(child)


def ingest_files(
    root: Path,
    target_extensions: Optional[Iterable[str]] = None,
    log_checksums: bool = False,
) -> Generator[tuple[Path, Optional[str]], None, None]:
    """
    Traverse root, recursively extract ZIPs, and yield target files with optional checksum.

    Yields tuples of (path, checksum or None).
    """
    exts = {e.lower() for e in (target_extensions or TARGET_EXTENSIONS)}
    for fpath in _iter_files_recursive(root):
        if fpath.suffix.lower() in exts:
            checksum = sha256sum(fpath) if log_checksums else None
            yield fpath, checksum


if __name__ == "__main__":
    root_dir = Path(".")
    for path, chksum in ingest_files(root_dir, log_checksums=False):
        print(path, chksum or "")
