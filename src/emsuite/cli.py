import sys
import argparse
from pathlib import Path
from .core import print_startup_message
from .tuning import main as tuning_main

def main():
    parser = argparse.ArgumentParser(print_startup_message())
    parser.add_argument("input_file", help="Input file (e.g., tuning.in)")
    
    args = parser.parse_args()
    
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file '{input_path}' not found")
        sys.exit(1)
    
    # Execute tuning module with the input file
    tuning_main(str(input_path))

if __name__ == "__main__":
    main()