"""Utility to obfuscate sensitive identifiers and remap locations to Alberta."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd


ALBERTA_CITIES: Tuple[Tuple[str, float, float], ...] = (
    ("Calgary", 51.0447, -114.0719),
    ("Edmonton", 53.5461, -113.4938),
    ("Red Deer", 52.2681, -113.8112),
    ("Lethbridge", 49.6956, -112.8450),
    ("St. Albert", 53.6305, -113.6256),
    ("Medicine Hat", 50.0405, -110.6765),
    ("Grande Prairie", 55.1707, -118.7947),
    ("Airdrie", 51.2917, -114.0144),
    ("Spruce Grove", 53.5464, -113.9187),
    ("Leduc", 53.2597, -113.5569),
    ("Okotoks", 50.7240, -113.9720),
    ("Fort McMurray", 56.7260, -111.3790),
    ("Banff", 51.1784, -115.5708),
    ("Canmore", 51.0890, -115.3595),
    ("Brooks", 50.5649, -111.8980),
    ("Camrose", 53.0167, -112.8200),
    ("Chestermere", 51.0379, -113.8225),
    ("Lloydminster", 53.2784, -110.0054),
    ("Sherwood Park", 53.5444, -113.3198),
    ("Wetaskiwin", 52.9692, -113.3761),
)


def hashed_token(value: str, prefix: str, length: int = 8) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest().upper()
    return f"{prefix}{digest[:length]}"


def hashed_postal(value: str) -> str:
    if not value:
        return ""
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    digits = "0123456789"
    return "".join(
        [
            "T",
            digits[int(digest[0], 16) % 10],
            letters[int(digest[1], 16) % 26],
            digits[int(digest[2], 16) % 10],
            letters[int(digest[3], 16) % 26],
            digits[int(digest[4], 16) % 10],
        ]
    )

def build_city_mapping(values: Iterable[str]) -> Dict[str, Tuple[str, float, float]]:
    mapping: Dict[str, Tuple[str, float, float]] = {}
    idx = 0
    for raw in values:
        key = raw.strip().lower()
        if not key or key == "online":
            continue
        if key not in mapping:
            mapping[key] = ALBERTA_CITIES[idx % len(ALBERTA_CITIES)]
            idx += 1
    return mapping
