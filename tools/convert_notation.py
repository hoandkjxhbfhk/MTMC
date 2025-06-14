import argparse
import re

def convert_number_format(number_str):
    """
    Converts a number string from potential scientific notation to standard decimal format.
    If the number is an integer, it's formatted as an integer.
    """
    try:
        f = float(number_str)
        if f == int(f):
            return str(int(f))
        else:
            # Format with high precision and remove trailing zeros
            return f"{f:.20f}".rstrip('0').rstrip('.')
    except (ValueError, TypeError):
        # Return the original string if it's not a number
        return number_str

def process_file(input_path, output_path):
    """
    Reads a file, converts all numbers to non-scientific notation, removes the last column,
    and saves to a new file.
    Handles space-separated values.
    """
    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line in infile:
            parts = line.strip().split()
            # Remove the last column if the line is not empty
            if parts:
                parts = parts[:-1]
            converted_parts = [convert_number_format(part) for part in parts]
            outfile.write(" ".join(converted_parts) + "\n")

def main():
    parser = argparse.ArgumentParser(description="Convert numbers in a text file from scientific notation to standard format and remove the last column.")
    parser.add_argument("input_file", help="The path to the input file.")
    parser.add_argument("output_file", help="The path to save the converted file.")
    args = parser.parse_args()

    try:
        process_file(args.input_file, args.output_file)
        print(f"File successfully converted and saved to {args.output_file}")
    except FileNotFoundError:
        print(f"Error: Input file not found at {args.input_file}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main() 