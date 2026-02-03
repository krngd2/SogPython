"""
SogPython CLI - Convert Gaussian Splat PLY files to PlayCanvas SOG format.

Usage:
    python -m SogPython input.ply output.sog [--iterations N]
"""

import argparse
import sys

from .sog_writer import convert_ply_to_sog


def main():
    parser = argparse.ArgumentParser(
        description='Convert Gaussian Splat PLY files to PlayCanvas SOG format.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
    python -m SogPython input.ply output.sog
    python -m SogPython model.ply model.sog --iterations 20
    python -m SogPython splat.ply splat.sog --no-bundle
        '''
    )
    
    parser.add_argument('input', help='Input PLY file path')
    parser.add_argument('output', help='Output SOG file path')
    parser.add_argument(
        '--iterations', '-i',
        type=int,
        default=10,
        help='Number of k-means iterations (default: 10)'
    )
    parser.add_argument(
        '--no-bundle',
        action='store_true',
        help='Write separate files instead of ZIP bundle'
    )
    
    args = parser.parse_args()
    
    try:
        convert_ply_to_sog(
            args.input,
            args.output,
            iterations=args.iterations,
            bundle=not args.no_bundle
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
