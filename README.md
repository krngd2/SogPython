# SogPy

Convert Gaussian Splat `.ply` files to PlayCanvas `.sog` format.

> ⚠️ **Note:** This is a direct Python conversion of [splat-transform](https://github.com/playcanvas/splat-transform)'s PLY to SOG Node.js code. One-shotted with Claude Opus 4.5 — use with caution! Contributions are very welcome.

## Installation

```bash
pip install sogpy
```

## Usage

### Command Line

```bash
# Basic conversion
sogpy input.ply output.sog

# With more clustering iterations (better quality, slower)
sogpy input.ply output.sog --iterations 20
```

### Python API

```python
from SogPython import convert_ply_to_sog

convert_ply_to_sog("input.ply", "output.sog")
```

## What is SOG?

SOG (Splat Optimized Graphics) is PlayCanvas's format for Gaussian splat data. It uses:

- **WebP lossless compression** for textures
- **K-means clustering** to reduce color and scale data to 256 values
- **Morton ordering** for spatial coherence
- **Smallest-three quaternion encoding** for rotations

This typically achieves **5-6x compression** compared to raw PLY files while preserving all data including spherical harmonics.

## Features

- ✅ Full PLY format support (binary little-endian)
- ✅ Spherical harmonics (bands 1-3)
- ✅ WebP lossless compression
- ✅ 16-bit position precision with log transform
- ✅ Smallest-three quaternion compression

## Requirements

- Python 3.8+
- numpy
- scipy
- Pillow

## License

MIT
