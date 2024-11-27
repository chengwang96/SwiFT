import argparse
import py7zr

def main():
    parser = argparse.ArgumentParser(description="Extract a 7z archive.")
    parser.add_argument('--input_file', help='Path to the 7z file.')
    parser.add_argument('--output_directory', help='Directory to extract files to.')

    args = parser.parse_args()

    print()
    with py7zr.SevenZipFile(args.input_file, mode='r') as archive:
        archive.extractall(path=args.output_directory)

if __name__ == "__main__":
    main()
