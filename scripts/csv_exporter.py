"""Exports a single CSV field to individual .txt files with 80-char line wrapping,
one file per row, saved alongside the source CSV.
Usage: python scripts/csv_exporter.py --db <path.csv> --field <column_name> --output <dir>
"""
import argparse
import csv
import sys
import re
from pathlib import Path

def wrap_text(text, width=80):
    """
    Inserts a newline after the first whitespace found 
    at or after the specified width.
    """
    if not text or len(text) <= width:
        return text
    
    result = []
    start = 0
    
    while start < len(text):
        # If remaining text is short, just add it and finish
        if len(text) - start <= width:
            result.append(text[start:])
            break
            
        # Find the first whitespace at or after (start + width)
        # We use a regex to find any whitespace character
        match = re.search(r'\s', text[start + width:])
        
        if match:
            # Calculate the actual index in the original string
            break_point = start + width + match.start()
            # Append the slice plus a newline
            result.append(text[start:break_point].strip())
            # Move the start pointer to after the whitespace
            start = break_point + 1
        else:
            # No whitespace found after the limit, append the rest
            result.append(text[start:])
            break
            
    return "\n".join(result)

def export_field_to_files(csv_path, field_name, output_dirname):
    db_file = Path(csv_path).resolve()
    
    if not db_file.exists():
        print(f"Error: File '{db_file}' not found.")
        sys.exit(1)

    output_dir = db_file.parent / output_dirname
    
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(db_file, mode='r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            
            if field_name not in reader.fieldnames:
                print(f"Error: Field '{field_name}' not in CSV.")
                sys.exit(1)
            
            for i, row in enumerate(reader, start=1):
                content = row[field_name]
                # Apply the custom line wrapping
                formatted_content = wrap_text(str(content), width=80)
                
                file_path = output_dir / f"entry_{i}.txt"
                with open(file_path, 'w', encoding='utf-8') as out_f:
                    out_f.write(formatted_content)
        
        print(f"Success! Files exported with 80-char wrap to: {output_dir}")

    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export CSV fields to individual text files.")
    parser.add_argument("--db", required=True, help="Path to the source CSV file")
    parser.add_argument("--field", required=True, help="The column name to export")
    parser.add_argument("--output", required=True, help="Name of the output folder (created next to CSV)")

    args = parser.parse_args()
    export_field_to_files(args.db, args.field, args.output)