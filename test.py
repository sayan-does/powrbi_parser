#!/usr/bin/env python3
"""
File Structure Generator Script

This script generates a tree-like representation of the directory structure
starting from a specified path. It can be used to document project organization.

Usage:
    python get_file_structure.py [directory_path] [--output output_file] [--exclude pattern1,pattern2]
    
Examples:
    python get_file_structure.py                    # Current directory
    python get_file_structure.py /path/to/project   # Specific directory
    python get_file_structure.py --output structure.txt  # Save to file
    python get_file_structure.py --exclude __pycache__,venv  # Exclude directories
"""

import os
import argparse
import fnmatch
import sys
from datetime import datetime


def get_file_structure(directory, prefix="", exclude_patterns=None):
    """
    Recursively generate a tree-like representation of the directory structure.
    
    Args:
        directory (str): The directory path to analyze
        prefix (str): Prefix for the current line (used for recursion)
        exclude_patterns (list): List of patterns to exclude
        
    Returns:
        str: The tree representation of the directory
    """
    if exclude_patterns is None:
        exclude_patterns = []

    result = []

    # Get all entries in the directory
    try:
        entries = os.listdir(directory)
    except PermissionError:
        return [f"{prefix}[Permission Denied]"]
    except FileNotFoundError:
        return [f"{prefix}[Directory Not Found]"]

    # Sort entries (directories first, then files)
    dirs = []
    files = []

    for entry in entries:
        # Check if entry should be excluded
        if any(fnmatch.fnmatch(entry, pattern) for pattern in exclude_patterns):
            continue

        full_path = os.path.join(directory, entry)
        if os.path.isdir(full_path):
            dirs.append(entry)
        else:
            files.append(entry)

    dirs.sort()
    files.sort()

    # Process all directories
    for i, dir_name in enumerate(dirs):
        is_last_dir = (i == len(dirs) - 1 and len(files) == 0)

        if is_last_dir:
            result.append(f"{prefix}└── {dir_name}/")
            sub_prefix = f"{prefix}    "
        else:
            result.append(f"{prefix}├── {dir_name}/")
            sub_prefix = f"{prefix}│   "

        full_path = os.path.join(directory, dir_name)
        result.extend(get_file_structure(
            full_path, sub_prefix, exclude_patterns))

    # Process all files
    for i, file_name in enumerate(files):
        if i == len(files) - 1:
            result.append(f"{prefix}└── {file_name}")
        else:
            result.append(f"{prefix}├── {file_name}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Generate a tree-like representation of a directory structure")
    parser.add_argument("directory", nargs="?", default=".",
                        help="Directory to analyze (default: current directory)")
    parser.add_argument(
        "--output", "-o", help="Output file (default: print to stdout)")
    parser.add_argument("--exclude", "-e", default="",
                        help="Comma-separated list of patterns to exclude")

    args = parser.parse_args()

    directory = os.path.abspath(args.directory)
    exclude_patterns = [p.strip()
                        for p in args.exclude.split(",") if p.strip()]

    # Common patterns to exclude by default
    default_excludes = ["__pycache__", "*.pyc", ".git",
                        ".DS_Store", "node_modules", ".idea", ".vscode"]
    exclude_patterns.extend(default_excludes)

    # Generate the structure
    structure = [f"Directory structure for: {directory}"]
    structure.append(
        f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    structure.append("")
    structure.extend(get_file_structure(
        directory, exclude_patterns=exclude_patterns))

    output = "\n".join(structure)

    # Output results
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"File structure written to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
