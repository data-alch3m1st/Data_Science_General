"""
print_tree.py
Reusable, dependency-free directory tree visualizer — importable or CLI-runnable.

Location convention: keep this in a shared utility folder (e.g.
/Users/t4ng0_br4v0/_jupyt3rNotebooks/_utility_scripts/) and import it
from any project notebook by adding that folder to sys.path.
"""

import os
import argparse

DEFAULT_IGNORE = {".git", "__pycache__", ".ipynb_checkpoints", ".DS_Store", ".venv"}


def print_tree(path, max_depth=None, show_hidden=False, ignore=None, show_files=True):
    """
    Print a hierarchical, box-drawing-style view of a directory tree.

    Args:
        path (str): Root directory to visualize.
        max_depth (int or None): Limit recursion depth (None = unlimited).
        show_hidden (bool): Include dotfiles/dotfolders not in the ignore set.
        ignore (set or None): Additional folder/file names to skip.
        show_files (bool): If False, only directories are shown.

    Returns:
        str: The full tree text (also printed to stdout).
    """
    ignore_set = set(DEFAULT_IGNORE)
    if ignore:
        ignore_set |= set(ignore)

    root = os.path.abspath(path)
    lines = [os.path.basename(root.rstrip(os.sep)) or root]

    def _walk(current_path, prefix, depth):
        if max_depth is not None and depth > max_depth:
            return
        try:
            entries = sorted(os.listdir(current_path))
        except PermissionError:
            return

        entries = [e for e in entries if e not in ignore_set]
        if not show_hidden:
            entries = [e for e in entries if not e.startswith(".")]
        if not show_files:
            entries = [e for e in entries
                       if os.path.isdir(os.path.join(current_path, e))]

        for i, entry in enumerate(entries):
            full_entry = os.path.join(current_path, entry)
            is_last = (i == len(entries) - 1)
            connector = "└── " if is_last else "├── "
            lines.append(f"{prefix}{connector}{entry}")

            if os.path.isdir(full_entry):
                extension = "    " if is_last else "│   "
                _walk(full_entry, prefix + extension, depth + 1)

    _walk(root, "", 1)
    tree_text = "\n".join(lines)
    print(tree_text)
    return tree_text


def _cli():
    parser = argparse.ArgumentParser(description="Print a directory tree.")
    parser.add_argument("path", nargs="?", default=".", help="Root path to visualize")
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--show-hidden", action="store_true")
    parser.add_argument("--dirs-only", action="store_true")
    args = parser.parse_args()

    print_tree(
        args.path,
        max_depth=args.max_depth,
        show_hidden=args.show_hidden,
        show_files=not args.dirs_only,
    )


if __name__ == "__main__":
    _cli()
