"""Nile-ViT: multimodal compound climate-extreme detection."""

# Must be the very first import: fixes PROJ paths on Windows systems
# with conflicting PostgreSQL/PostGIS installations. See nilevit/_environment.py.
from nilevit import _environment  # noqa: F401

__version__ = "0.1.0"
__all__ = ["__version__"]
