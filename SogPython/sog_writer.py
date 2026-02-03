"""
SOG (Splat Optimized Graphics) format writer.

Converts Gaussian Splat PLY data to PlayCanvas SOG format using:
- WebP lossless compression for textures
- K-means clustering for scale and color data
- Morton ordering for spatial coherence
"""

import json
import math
import zipfile
import io
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import numpy as np
from PIL import Image

from .ply_reader import read_ply
from .morton_order import sort_morton_order
from .kmeans import cluster_1d, cluster_sh
from .utils import sigmoid, log_transform


# Spherical harmonics coefficient names
SH_NAMES = [f"f_rest_{i}" for i in range(45)]


def encode_webp_lossless(data: np.ndarray, width: int, height: int) -> bytes:
    """Encode RGBA data as lossless WebP."""
    # Ensure data is the right shape and type
    rgba = data.reshape(height, width, 4).astype(np.uint8)
    image = Image.fromarray(rgba, mode='RGBA')
    
    buffer = io.BytesIO()
    image.save(buffer, format='WEBP', lossless=True)
    return buffer.getvalue()


def calc_min_max(data: Dict[str, np.ndarray], column_names: List[str], indices: np.ndarray) -> List[Tuple[float, float]]:
    """Calculate min/max for specified columns using only indexed rows."""
    result = []
    for name in column_names:
        values = data[name][indices]
        result.append((float(values.min()), float(values.max())))
    return result


def write_means(data: Dict[str, np.ndarray], indices: np.ndarray, width: int, height: int) -> Tuple[bytes, bytes, Dict]:
    """
    Write position data as two 16-bit WebP textures (low and high bytes).
    
    Uses log transform on positions for better precision distribution.
    """
    n = len(indices)
    channels = 4
    
    means_l = np.zeros(width * height * channels, dtype=np.uint8)
    means_u = np.zeros(width * height * channels, dtype=np.uint8)
    
    # Calculate min/max with log transform
    column_names = ['x', 'y', 'z']
    min_max_raw = calc_min_max(data, column_names, indices)
    min_max = [(log_transform(np.array([mm[0]]))[0], log_transform(np.array([mm[1]]))[0]) for mm in min_max_raw]
    
    x_data = data['x'][indices]
    y_data = data['y'][indices]
    z_data = data['z'][indices]
    
    for i in range(n):
        # Apply log transform and normalize to 16-bit
        x_log = log_transform(np.array([x_data[i]]))[0]
        y_log = log_transform(np.array([y_data[i]]))[0]
        z_log = log_transform(np.array([z_data[i]]))[0]
        
        x_range = min_max[0][1] - min_max[0][0]
        y_range = min_max[1][1] - min_max[1][0]
        z_range = min_max[2][1] - min_max[2][0]
        
        x_norm = 65535 * (x_log - min_max[0][0]) / x_range if x_range > 0 else 0
        y_norm = 65535 * (y_log - min_max[1][0]) / y_range if y_range > 0 else 0
        z_norm = 65535 * (z_log - min_max[2][0]) / z_range if z_range > 0 else 0
        
        x_int = int(max(0, min(65535, x_norm)))
        y_int = int(max(0, min(65535, y_norm)))
        z_int = int(max(0, min(65535, z_norm)))
        
        ti = i  # layout is identity
        
        # Low bytes
        means_l[ti * 4] = x_int & 0xff
        means_l[ti * 4 + 1] = y_int & 0xff
        means_l[ti * 4 + 2] = z_int & 0xff
        means_l[ti * 4 + 3] = 0xff
        
        # High bytes
        means_u[ti * 4] = (x_int >> 8) & 0xff
        means_u[ti * 4 + 1] = (y_int >> 8) & 0xff
        means_u[ti * 4 + 2] = (z_int >> 8) & 0xff
        means_u[ti * 4 + 3] = 0xff
    
    webp_l = encode_webp_lossless(means_l, width, height)
    webp_u = encode_webp_lossless(means_u, width, height)
    
    return webp_l, webp_u, {
        'mins': [mm[0] for mm in min_max],
        'maxs': [mm[1] for mm in min_max]
    }


def write_quaternions(data: Dict[str, np.ndarray], indices: np.ndarray, width: int, height: int) -> bytes:
    """
    Write quaternion rotation data using smallest-three compression.
    
    Stores three components (excluding the largest) in range [-1, 1] scaled by sqrt(2).
    The alpha channel encodes which component was dropped (252-255).
    """
    n = len(indices)
    channels = 4
    quats = np.zeros(width * height * channels, dtype=np.uint8)
    
    rot_0 = data['rot_0'][indices]
    rot_1 = data['rot_1'][indices]
    rot_2 = data['rot_2'][indices]
    rot_3 = data['rot_3'][indices]
    
    sqrt2 = math.sqrt(2)
    
    for i in range(n):
        q = [rot_0[i], rot_1[i], rot_2[i], rot_3[i]]
        
        # Normalize
        length = math.sqrt(sum(c * c for c in q))
        if length > 0:
            q = [c / length for c in q]
        
        # Find largest component
        max_comp = max(range(4), key=lambda j: abs(q[j]))
        
        # Ensure largest component is positive
        if q[max_comp] < 0:
            q = [-c for c in q]
        
        # Scale by sqrt(2)
        q = [c * sqrt2 for c in q]
        
        # Get indices of the three smallest components
        idx_map = {
            0: [1, 2, 3],
            1: [0, 2, 3],
            2: [0, 1, 3],
            3: [0, 1, 2]
        }
        idx = idx_map[max_comp]
        
        ti = i
        quats[ti * 4] = int(255 * (q[idx[0]] * 0.5 + 0.5))
        quats[ti * 4 + 1] = int(255 * (q[idx[1]] * 0.5 + 0.5))
        quats[ti * 4 + 2] = int(255 * (q[idx[2]] * 0.5 + 0.5))
        quats[ti * 4 + 3] = 252 + max_comp
    
    return encode_webp_lossless(quats, width, height)


def write_scales(data: Dict[str, np.ndarray], indices: np.ndarray, width: int, height: int, iterations: int) -> Tuple[bytes, List[float]]:
    """
    Write scale data using k-means clustering to 256 values.
    """
    n = len(indices)
    channels = 4
    
    # Cluster scale data
    column_names = ['scale_0', 'scale_1', 'scale_2']
    centroids, labels_dict = cluster_1d(data, column_names, 256, iterations)
    
    # Build output texture
    scales_buf = np.zeros(width * height * channels, dtype=np.uint8)
    
    for i, idx in enumerate(indices):
        ti = i
        scales_buf[ti * 4] = labels_dict['scale_0'][idx]
        scales_buf[ti * 4 + 1] = labels_dict['scale_1'][idx]
        scales_buf[ti * 4 + 2] = labels_dict['scale_2'][idx]
        scales_buf[ti * 4 + 3] = 255
    
    return encode_webp_lossless(scales_buf, width, height), centroids.tolist()


def write_colors(data: Dict[str, np.ndarray], indices: np.ndarray, width: int, height: int, iterations: int) -> Tuple[bytes, List[float]]:
    """
    Write color (DC spherical harmonic) and opacity data.
    
    Colors are clustered to 256 values, opacity is stored as sigmoid(raw) in alpha channel.
    """
    n = len(indices)
    channels = 4
    
    # Cluster color data
    column_names = ['f_dc_0', 'f_dc_1', 'f_dc_2']
    centroids, labels_dict = cluster_1d(data, column_names, 256, iterations)
    
    # Calculate sigmoid of opacity
    opacity_raw = data['opacity']
    opacity_sigmoid = sigmoid(opacity_raw)
    opacity_u8 = np.clip(opacity_sigmoid * 255, 0, 255).astype(np.uint8)
    
    # Build output texture
    sh0_buf = np.zeros(width * height * channels, dtype=np.uint8)
    
    for i, idx in enumerate(indices):
        ti = i
        sh0_buf[ti * 4] = labels_dict['f_dc_0'][idx]
        sh0_buf[ti * 4 + 1] = labels_dict['f_dc_1'][idx]
        sh0_buf[ti * 4 + 2] = labels_dict['f_dc_2'][idx]
        sh0_buf[ti * 4 + 3] = opacity_u8[idx]
    
    return encode_webp_lossless(sh0_buf, width, height), centroids.tolist()


def write_sh(data: Dict[str, np.ndarray], indices: np.ndarray, width: int, height: int, 
             sh_bands: int, iterations: int) -> Optional[Dict]:
    """
    Write higher-order spherical harmonics data.
    
    Uses two-level clustering:
    1. Cluster all SH points into palette_size centroids
    2. Quantize centroid values using 256-value codebook
    """
    sh_coeffs = {1: 3, 2: 8, 3: 15}[sh_bands]
    sh_column_names = SH_NAMES[:sh_coeffs * 3]
    
    # Check if all SH columns exist
    for name in sh_column_names:
        if name not in data:
            return None
    
    n = len(indices)
    palette_size = min(64 * 1024, max(1024, 2 ** int(math.log2(n / 1024)) * 1024))
    
    # Two-level clustering
    labels, centroids_quantized, codebook = cluster_sh(data, sh_column_names, palette_size, iterations)
    
    # Write centroids texture
    n_centroids = len(centroids_quantized)
    centroid_width = 64 * sh_coeffs
    centroid_height = int(math.ceil(n_centroids / 64))
    
    centroids_buf = np.zeros(centroid_width * centroid_height * 4, dtype=np.uint8)
    
    for i, row in enumerate(centroids_quantized):
        for j in range(sh_coeffs):
            # R, G, B channels for each coefficient
            if j * 3 < len(row):
                centroids_buf[i * sh_coeffs * 4 + j * 4 + 0] = row[j * 3] if j * 3 < len(row) else 0
                centroids_buf[i * sh_coeffs * 4 + j * 4 + 1] = row[j * 3 + 1] if j * 3 + 1 < len(row) else 0
                centroids_buf[i * sh_coeffs * 4 + j * 4 + 2] = row[j * 3 + 2] if j * 3 + 2 < len(row) else 0
            centroids_buf[i * sh_coeffs * 4 + j * 4 + 3] = 0xff
    
    centroids_webp = encode_webp_lossless(centroids_buf, centroid_width, centroid_height)
    
    # Write labels texture (16-bit indices)
    labels_buf = np.zeros(width * height * 4, dtype=np.uint8)
    
    for i, idx in enumerate(indices):
        label = labels[idx]
        ti = i
        labels_buf[ti * 4 + 0] = label & 0xff
        labels_buf[ti * 4 + 1] = (label >> 8) & 0xff
        labels_buf[ti * 4 + 2] = 0
        labels_buf[ti * 4 + 3] = 0xff
    
    labels_webp = encode_webp_lossless(labels_buf, width, height)
    
    return {
        'count': palette_size,
        'bands': sh_bands,
        'codebook': codebook.tolist(),
        'files': ['shN_centroids.webp', 'shN_labels.webp'],
        'centroids_webp': centroids_webp,
        'labels_webp': labels_webp
    }


def detect_sh_bands(data: Dict[str, np.ndarray]) -> int:
    """Detect how many spherical harmonics bands are present in the data."""
    # Band 1: f_rest_0 to f_rest_8 (9 coeffs)
    # Band 2: f_rest_0 to f_rest_23 (24 coeffs)
    # Band 3: f_rest_0 to f_rest_44 (45 coeffs)
    
    for name in SH_NAMES[:9]:
        if name not in data:
            return 0
    # Has at least band 1
    
    for name in SH_NAMES[9:24]:
        if name not in data:
            return 1
    # Has at least band 2
    
    for name in SH_NAMES[24:45]:
        if name not in data:
            return 2
    
    return 3


def convert_ply_to_sog(input_path: str, output_path: str, iterations: int = 10, bundle: bool = True) -> None:
    """
    Convert a Gaussian Splat PLY file to PlayCanvas SOG format.
    
    Args:
        input_path: Path to input .ply file
        output_path: Path to output .sog file
        iterations: Number of k-means iterations (default 10)
        bundle: If True, create a ZIP bundle (default). If False, write separate files.
    """
    print(f"Reading PLY file: {input_path}")
    data = read_ply(input_path)
    
    num_rows = len(data['x'])
    print(f"Loaded {num_rows} splats")
    
    # Generate Morton-ordered indices
    print("Generating Morton order...")
    indices = sort_morton_order(data['x'], data['y'], data['z'])
    
    # Calculate texture dimensions (multiples of 4)
    width = int(math.ceil(math.sqrt(num_rows) / 4)) * 4
    height = int(math.ceil(num_rows / width / 4)) * 4
    print(f"Texture size: {width}x{height}")
    
    # Prepare output
    output_files = {}
    
    # Write positions
    print("Writing positions...")
    means_l, means_u, means_info = write_means(data, indices, width, height)
    output_files['means_l.webp'] = means_l
    output_files['means_u.webp'] = means_u
    
    # Write quaternions
    print("Writing quaternions...")
    quats_webp = write_quaternions(data, indices, width, height)
    output_files['quats.webp'] = quats_webp
    
    # Write scales
    print("Compressing scales...")
    scales_webp, scales_codebook = write_scales(data, indices, width, height, iterations)
    output_files['scales.webp'] = scales_webp
    
    # Write colors
    print("Compressing colors...")
    sh0_webp, colors_codebook = write_colors(data, indices, width, height, iterations)
    output_files['sh0.webp'] = sh0_webp
    
    # Write spherical harmonics if present
    sh_bands = detect_sh_bands(data)
    sh_info = None
    if sh_bands > 0:
        print(f"Compressing spherical harmonics (bands={sh_bands})...")
        sh_result = write_sh(data, indices, width, height, sh_bands, iterations)
        if sh_result:
            output_files['shN_centroids.webp'] = sh_result['centroids_webp']
            output_files['shN_labels.webp'] = sh_result['labels_webp']
            sh_info = {
                'count': sh_result['count'],
                'bands': sh_result['bands'],
                'codebook': sh_result['codebook'],
                'files': sh_result['files']
            }
    
    # Build metadata
    print("Finalizing...")
    meta = {
        'version': 2,
        'asset': {
            'generator': 'SogPython v1.0.0'
        },
        'count': num_rows,
        'means': {
            'mins': means_info['mins'],
            'maxs': means_info['maxs'],
            'files': ['means_l.webp', 'means_u.webp']
        },
        'scales': {
            'codebook': scales_codebook,
            'files': ['scales.webp']
        },
        'quats': {
            'files': ['quats.webp']
        },
        'sh0': {
            'codebook': colors_codebook,
            'files': ['sh0.webp']
        }
    }
    
    if sh_info:
        meta['shN'] = sh_info
    
    meta_json = json.dumps(meta, separators=(',', ':')).encode('utf-8')
    
    if bundle:
        # Write as ZIP archive
        print(f"Writing SOG bundle: {output_path}")
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_STORED) as zf:
            zf.writestr('meta.json', meta_json)
            for filename, content in output_files.items():
                zf.writestr(filename, content)
    else:
        # Write separate files
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Write meta.json to the specified output path
        with open(output_path, 'wb') as f:
            f.write(meta_json)
        
        for filename, content in output_files.items():
            filepath = output_dir / filename
            with open(filepath, 'wb') as f:
                f.write(content)
            print(f"  Wrote: {filepath}")
    
    print("Done!")


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print("Usage: python -m SogPython.sog_writer input.ply output.sog")
        sys.exit(1)
    
    convert_ply_to_sog(sys.argv[1], sys.argv[2])
