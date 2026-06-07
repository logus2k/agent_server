# app/geoip.py
"""Optional GeoLite2-City IP geolocation for the admin Clients view.

Ported from avatar_server. We read the sapics ip-location-db mirror
(https://github.com/sapics/ip-location-db/tree/main/geolite2-city-mmdb),
which ships IPv4 and IPv6 as separate MMDB files. Drop either or both into
``app/data/geoip/`` (bind-mounted, so no image rebuild) to enable geo
enrichment. Without them, lookups return None and the Clients view shows
raw IPs only.

Everything here is best-effort: no function raises, so a missing DB, a
missing ``maxminddb`` library, or a malformed record degrades to None.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

# app/data/geoip/ — `data` is bind-mounted rw into the container, so the
# DBs live beside agent_config.json and are not baked into the image.
_GEOIP_DIR = Path(__file__).resolve().parent / "data" / "geoip"
_GEOIP_DB_V4 = _GEOIP_DIR / "geolite2-city-ipv4.mmdb"
_GEOIP_DB_V6 = _GEOIP_DIR / "geolite2-city-ipv6.mmdb"

_readers: Dict[str, Any] = {}        # {"v4": Reader, "v6": Reader}
_init_attempted = False

# Private / loopback / link-local prefixes that never geolocate. Includes the
# full 172.16/12 block (Docker bridge networks live here — e.g. 172.20.0.1).
_PRIVATE_PREFIXES = tuple(
    ["10.", "192.168.", "127.", "169.254.", "::1", "fc00:", "fd00:", "fe80:"]
    + [f"172.{i}." for i in range(16, 32)]
)


def db_present() -> Dict[str, bool]:
    return {
        "any": _GEOIP_DB_V4.exists() or _GEOIP_DB_V6.exists(),
        "v4": _GEOIP_DB_V4.exists(),
        "v6": _GEOIP_DB_V6.exists(),
    }


def _get_readers() -> Dict[str, Any]:
    """Lazy-open the GeoLite2 readers. Each file is independently optional.

    We use maxminddb directly (not geoip2.Reader.city) because the sapics
    mirror declares ``database_type = "city ipv4"`` instead of MaxMind's
    ``"GeoLite2-City"``; geoip2's strict type check rejects that, while
    maxminddb just returns the raw record dict."""
    global _init_attempted
    if _init_attempted:
        return _readers
    _init_attempted = True
    try:
        import maxminddb
    except Exception as e:  # noqa: BLE001
        print(f"[geoip] maxminddb not installed ({type(e).__name__}: {e}); "
              f"clients endpoint will return raw IPs only", flush=True)
        return _readers
    for kind, path in (("v4", _GEOIP_DB_V4), ("v6", _GEOIP_DB_V6)):
        if not path.exists():
            continue
        try:
            _readers[kind] = maxminddb.open_database(str(path))
            print(f"[geoip] GeoLite2 {kind} loaded from {path.name}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[geoip] GeoLite2 {kind} load failed ({type(e).__name__}: {e})",
                  flush=True)
    return _readers


def lookup(ip: Optional[str]) -> Optional[Dict[str, Any]]:
    """Geolocate one IP against GeoLite2 (sapics flattened schema). Returns
    None for private/loopback/invalid IPs or when no matching DB is loaded."""
    if not ip or ip in ("?", "localhost"):
        return None
    if ip.startswith(_PRIVATE_PREFIXES):
        return None
    readers = _get_readers()
    if not readers:
        return None
    reader = readers.get("v6" if ":" in ip else "v4")
    if reader is None:
        return None
    try:
        rec = reader.get(ip)
    except Exception:  # noqa: BLE001
        return None
    if not rec:
        return None

    def _s(key):
        v = rec.get(key)
        return v if v else None

    return {
        "country_code": _s("country_code"),
        "city": _s("city"),
        "region": _s("state1") or _s("state2"),
        "lat": rec.get("latitude"),
        "lon": rec.get("longitude"),
        "timezone": _s("timezone"),
    }
