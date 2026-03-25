# Entry point for data processing CLI

def main():

    import argparse
    import sys
    from preprocess.generate_dataset import main as generate_main

    parser = argparse.ArgumentParser(description="ShellGNN Data Processing CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: generate-dataset
    gen_parser = subparsers.add_parser("generate-dataset", help="Generate HDF5 dataset from raw files")
    gen_parser.add_argument("--base-dir", type=str, default="new_part_split_trias", help="Base directory with raw data")
    gen_parser.add_argument("--hdf5-name", type=str, default="new_part_split_trias", help="Output HDF5 file name (no extension)")
    gen_parser.add_argument("--log-name", type=str, default="new_part_split_trias", help="Log file name")

    args = parser.parse_args()

    if args.command == "generate-dataset":
        sys.exit(generate_main(args))

if __name__ == "__main__":
    main()
