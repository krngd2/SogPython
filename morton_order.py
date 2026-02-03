"""Morton order (Z-order curve) sorting for spatial coherence."""

import numpy as np
from typing import Optional


def part1by2(x: np.ndarray) -> np.ndarray:
    """Spread bits of x by inserting two zeros between each bit."""
    x = x.astype(np.uint32)
    x = x & 0x000003ff
    x = (x ^ (x << 16)) & 0xff0000ff
    x = (x ^ (x << 8)) & 0x0300f00f
    x = (x ^ (x << 4)) & 0x030c30c3
    x = (x ^ (x << 2)) & 0x09249249
    return x


def encode_morton3(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Encode 3D coordinates into Morton code."""
    return (part1by2(z) << 2) + (part1by2(y) << 1) + part1by2(x)


def sort_morton_order(x: np.ndarray, y: np.ndarray, z: np.ndarray, indices: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Sort points into Morton/Z-order for better spatial coherence.
    
    Args:
        x, y, z: Position coordinate arrays
        indices: Optional array of indices to sort (if None, creates 0..n-1)
        
    Returns:
        Sorted indices array
    """
    n = len(x)
    if indices is None:
        indices = np.arange(n, dtype=np.uint32)
    else:
        indices = indices.copy()
    
    _sort_morton_recursive(x, y, z, indices)
    return indices


def _sort_morton_recursive(x: np.ndarray, y: np.ndarray, z: np.ndarray, indices: np.ndarray) -> None:
    """Recursively sort indices by Morton code."""
    if len(indices) == 0:
        return
    
    # Get values for current indices
    cx = x[indices]
    cy = y[indices]
    cz = z[indices]
    
    # Calculate bounds
    mx, Mx = cx.min(), cx.max()
    my, My = cy.min(), cy.max()
    mz, Mz = cz.min(), cz.max()
    
    xlen = Mx - mx
    ylen = My - my
    zlen = Mz - mz
    
    # Check for invalid or identical points
    if not (np.isfinite(xlen) and np.isfinite(ylen) and np.isfinite(zlen)):
        return
    if xlen == 0 and ylen == 0 and zlen == 0:
        return
    
    # Normalize to 0-1023 range
    xmul = 1024 / xlen if xlen > 0 else 0
    ymul = 1024 / ylen if ylen > 0 else 0
    zmul = 1024 / zlen if zlen > 0 else 0
    
    ix = np.minimum(1023, ((cx - mx) * xmul)).astype(np.uint32)
    iy = np.minimum(1023, ((cy - my) * ymul)).astype(np.uint32)
    iz = np.minimum(1023, ((cz - mz) * zmul)).astype(np.uint32)
    
    # Compute Morton codes
    morton = encode_morton3(ix, iy, iz)
    
    # Sort by Morton code
    order = np.argsort(morton)
    indices[:] = indices[order]
    morton = morton[order]
    
    # Recursively sort large buckets (same Morton code)
    start = 0
    while start < len(indices):
        end = start + 1
        while end < len(indices) and morton[end] == morton[start]:
            end += 1
        
        # Only recurse for buckets larger than 256
        if end - start > 256:
            _sort_morton_recursive(x, y, z, indices[start:end])
        
        start = end
