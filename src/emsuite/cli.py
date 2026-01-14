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
    
    # calc_type.add_argument(
    #     '-p', '--potential',
    #     metavar='INPUT_FILE',
    #     help='Generate electrostatic potential map of chemical system'
    # )
    
    # calc_type.add_argument(
    #     '-c', '--combined',
    #     metavar='INPUT_FILE',
    #     help='Run combined potential-tuning analysis'
    # )
    
    args = parser.parse_args()
    
    # Route to appropriate channel
    if args.tuning:
        run_tuning(args.tuning)
    # elif args.potential:
    #     run_potential(args.potential)
    # elif args.combined:
    #     run_combined(args.combined)

def run_tuning(input_file: str):
    """Execute the tuning channel."""
    from .tuning import main as tuning_main
    
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"Error: Input file '{input_path}' not found")
        sys.exit(1)
    
    tuning_main(str(input_path))

# def run_potential(input_file: str):
#     """Execute the potential channel.
#     
#     Generates electrostatic potential maps for chemical systems.
#     """
#     from .potential import main as potential_main
#     
#     input_path = Path(input_file)
#     if not input_path.exists():
#         print(f"Error: Input file '{input_path}' not found")
#         sys.exit(1)
#     
#     potential_main(str(input_path))

# def run_combined(input_file: str):
#     """Execute the combined channel.
#     
#     Connects electrostatic potentials to theoretical tuning analysis.
#     """
#     from .combined import main as combined_main
#     
#     input_path = Path(input_file)
#     if not input_path.exists():
#         print(f"Error: Input file '{input_path}' not found")
#         sys.exit(1)
#     
#     combined_main(str(input_path))

if __name__ == "__main__":
    main()