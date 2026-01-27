import sys
import argparse
from pathlib import Path
from .core import print_startup_message

def main():
    # Print startup message before parsing
    print_startup_message()
    
    parser = argparse.ArgumentParser(
        prog='emsuite',
        description='EMSuite - Electrostatic Map Suite'
    )
    
    # Create mutually exclusive group for calculation types
    calc_type = parser.add_mutually_exclusive_group(required=True)
    
    calc_type.add_argument(
        '-t', '--tuning',
        metavar='INPUT_FILE',
        help='Run electrostatic tuning calculation'
    )
    
    calc_type.add_argument(
        '-s', '--surface',
        metavar='INPUT_FILE',
        help='Generate VDW surface from input file'
    )
    
    args = parser.parse_args()
    
    # Route to appropriate channel
    if args.tuning:
        run_tuning(args.tuning)
    elif args.surface:
        run_surface(args.surface)

def run_surface(input_file: str):
    """Execute the surface generation channel."""
    from .surface import run_surface_calculation
    
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"Error: Input file '{input_path}' not found")
        sys.exit(1)
    
    run_surface_calculation(str(input_path))

def run_tuning(input_file: str):
    """Execute the tuning channel."""
    from .tuning import main as tuning_main
    
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"Error: Input file '{input_path}' not found")
        sys.exit(1)
    
    tuning_main(str(input_path))

if __name__ == "__main__":
    main()