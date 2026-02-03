"""PLY file reader for Gaussian Splat data."""

import numpy as np
from typing import Dict, List, Tuple


def get_numpy_dtype(ply_type: str):
    """Map PLY type names to numpy dtypes."""
    type_map = {
        'char': np.int8,
        'uchar': np.uint8,
        'short': np.int16,
        'ushort': np.uint16,
        'int': np.int32,
        'uint': np.uint32,
        'float': np.float32,
        'double': np.float64,
    }
    return type_map.get(ply_type)


def parse_ply_header(data: bytes) -> Tuple[Dict, int]:
    """
    Parse PLY header and return element definitions and header end position.
    
    Returns:
        Tuple of (header_info, header_end_position)
        header_info = {'comments': [...], 'elements': [{'name': ..., 'count': ..., 'properties': [...]}]}
    """
    # Find end of header
    header_end = data.find(b'end_header\n')
    if header_end == -1:
        header_end = data.find(b'end_header\r\n')
        end_len = 12
    else:
        end_len = 11
    
    if header_end == -1:
        raise ValueError("Invalid PLY file: no end_header found")
    
    header_text = data[:header_end].decode('ascii')
    lines = [l.strip() for l in header_text.split('\n') if l.strip()]
    
    # Validate PLY magic
    if not lines[0].startswith('ply'):
        raise ValueError("Invalid PLY file: missing 'ply' header")
    
    header_info = {
        'comments': [],
        'elements': [],
        'format': 'binary_little_endian'
    }
    
    current_element = None
    
    for line in lines[1:]:
        words = line.split()
        
        if words[0] == 'format':
            header_info['format'] = words[1]
        elif words[0] == 'comment':
            header_info['comments'].append(' '.join(words[1:]))
        elif words[0] == 'element':
            if len(words) != 3:
                raise ValueError(f"Invalid element line: {line}")
            current_element = {
                'name': words[1],
                'count': int(words[2]),
                'properties': []
            }
            header_info['elements'].append(current_element)
        elif words[0] == 'property':
            if current_element is None or len(words) < 3:
                raise ValueError(f"Invalid property line: {line}")
            # Skip list properties (not used in splat data)
            if words[1] == 'list':
                continue
            dtype = get_numpy_dtype(words[1])
            if dtype is None:
                raise ValueError(f"Unknown property type: {words[1]}")
            current_element['properties'].append({
                'name': words[2],
                'type': words[1],
                'dtype': dtype
            })
    
    return header_info, header_end + end_len


def read_ply(filepath: str) -> Dict[str, np.ndarray]:
    """
    Read a PLY file and return a dictionary of property arrays.
    
    Args:
        filepath: Path to the PLY file
        
    Returns:
        Dictionary mapping property names to numpy arrays
    """
    with open(filepath, 'rb') as f:
        data = f.read()
    
    header, data_start = parse_ply_header(data)
    
    # Only support binary little endian format
    if 'binary_little_endian' not in header['format']:
        raise ValueError(f"Unsupported PLY format: {header['format']}. Only binary_little_endian is supported.")
    
    result = {}
    offset = data_start
    
    for element in header['elements']:
        name = element['name']
        count = element['count']
        properties = element['properties']
        
        if not properties:
            continue
        
        # Build structured dtype for this element
        dtype_list = [(p['name'], p['dtype']) for p in properties]
        element_dtype = np.dtype(dtype_list)
        
        # Read element data
        element_size = element_dtype.itemsize
        element_data = np.frombuffer(
            data, 
            dtype=element_dtype, 
            count=count, 
            offset=offset
        )
        offset += element_size * count
        
        # Store vertex element properties directly
        if name == 'vertex':
            for prop in properties:
                result[prop['name']] = element_data[prop['name']].astype(np.float32)
    
    return result


def get_splat_data(ply_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Organize PLY data into splat data structure.
    
    Returns dictionary with:
        - x, y, z: positions
        - rot_0, rot_1, rot_2, rot_3: quaternion rotation
        - scale_0, scale_1, scale_2: scale (log)
        - f_dc_0, f_dc_1, f_dc_2: DC spherical harmonic (color)
        - opacity: opacity (logit)
        - f_rest_0 to f_rest_44: higher order SH (optional)
    """
    return ply_data
