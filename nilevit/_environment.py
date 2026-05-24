"""Environment fixes that must run before any geospatial imports.

Windows-specific PROJ pitfalls this module guards against:

1. **System-wide PROJ leak.** A separate PROJ install (commonly from
   PostgreSQL/PostGIS) exposes an old proj.db via global PROJ_LIB /
   PROJ_DATA env vars or default search paths.

2. **Bundled-version mismatch inside the venv.** Each Python geospatial
   wheel (pyproj, rasterio, fiona) ships its own PROJ C library and
   matching proj.db. If pyproj ships PROJ 9.3 (db layout v4) and
   rasterio ships PROJ 9.5 (db layout v5+), pointing PROJ_DATA at the
   wrong package's proj.db triggers a CRSError on first EPSG lookup.

Strategy: locate the bundled proj.db on disk *without* importing the
geospatial packages yet, preferring rasterio's data dir because
rasterio is what triggers the actual PROJ call in our stack
(terratorch -> torchgeo -> rasterio -> PROJ).
"""

from __future__ import annotations

import os
import sysconfig
from pathlib import Path


def fix_proj_paths() -> None:
    """Point PROJ_DATA / PROJ_LIB / GDAL_DATA at venv-bundled data dirs."""
    site_packages = Path(sysconfig.get_path("purelib"))

    # rasterio first: it's the one actually calling PROJ in our stack.
    for pkg_name in ("rasterio", "pyproj"):
        pkg_root = site_packages / pkg_name
        if not pkg_root.is_dir():
            continue

        for subpath in (
            "proj_data",  # rasterio's typical location
            "proj_dir/share/proj",  # pyproj's typical location
            "proj/share/proj",
            "share/proj",
        ):
            proj_dir = pkg_root / subpath
            if (proj_dir / "proj.db").is_file():
                os.environ["PROJ_DATA"] = str(proj_dir)
                os.environ["PROJ_LIB"] = str(proj_dir)

                # Best-effort: set GDAL_DATA from the same package.
                for gdal_sub in ("gdal_data", "share/gdal"):
                    gdal_dir = pkg_root / gdal_sub
                    if gdal_dir.is_dir():
                        os.environ["GDAL_DATA"] = str(gdal_dir)
                        break
                return


fix_proj_paths()
