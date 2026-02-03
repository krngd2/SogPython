"""
SogPython - Convert Gaussian Splat PLY files to PlayCanvas SOG format.

Usage:
    from SogPython import convert_ply_to_sog
    convert_ply_to_sog("input.ply", "output.sog")
"""

from .sog_writer import convert_ply_to_sog
from .ply_reader import read_ply

__version__ = "1.0.0"
__all__ = ["convert_ply_to_sog", "read_ply"]
