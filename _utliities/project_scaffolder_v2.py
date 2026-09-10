"""
project_scaffolder.py
Reusable project directory scaffolder — importable or CLI-runnable.

Location convention: keep this in a shared utility folder (e.g.
/Users/t4ng0_br4v0/_jupyt3rNotebooks/_utility_scripts/) and import it
from any project notebook by adding that folder to sys.path.
"""

import os
import argparse

PRESETS = {
    "ml-project": [
        "benchmarks/results",
        "data/raw/real_images",
        "data/raw/synthetic_images",
        "data/ground_truth",
        "data/interim",
        "data/processed",
        "models",
        "notebooks",
        "src/ocr",
        "src/validation",
        "src/pipeline",
        "src/utils",
        "scripts",
        "tests",
        "docs",
        "output",
    ],
    "ds-standard": [
        "data/raw",
        "data/interim",
        "data/processed",
        "notebooks",
        "src",
        "reports/figures",
        "output",
    ],
    "minimal": [
        "data",
        "notebooks",
        "src",
        "output",
    ],
}

GITIGNORE_TEMPLATE = """# Data & models
data/raw/
data/interim/
data/processed/
models/
output/

# Notebooks checkpoints
.ipynb_checkpoints/

# Python
__pycache__/
*.pyc
.venv/
.env
"""

README_TEMPLATE = """# {name}

## Structure
This project was scaffolded with `project_scaffolder.py`.
See directory tree below (generate with `print_tree.py`).

## Notes
- Raw data lives in `data/raw/`, never modified in place.
- Ground truth / labels live in `data/ground_truth/`.
- Pipeline code lives in `src/`.
- Run outputs land in `output/` (gitignored).
"""


def scaffold_project(name, preset="ml-project", base_dir=".", create_gitignore=True,
                      create_readme=True, verbose=True):
    """
    Create a project directory tree from a named preset.

    Args:
        name (str): Project folder name (created under base_dir).
        preset (str): One of PRESETS.keys().
        base_dir (str): Parent directory in which to create the project.
        create_gitignore (bool): Write a starter .gitignore.
        create_readme (bool): Write a starter README.md.
        verbose (bool): Print progress.

    Returns:
        str: Absolute path to the created project root.

    Raises:
        ValueError: If preset name is not recognized.
    """
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset '{preset}'. Options: {list(PRESETS.keys())}")

    project_root = os.path.abspath(os.path.join(base_dir, name))
    os.makedirs(project_root, exist_ok=True)

    created, skipped = [], []
    for rel_path in PRESETS[preset]:
        full_path = os.path.join(project_root, rel_path)
        if os.path.exists(full_path):
            skipped.append(rel_path)
            continue
        os.makedirs(full_path, exist_ok=True)
        created.append(rel_path)

    _cleanup_and_placeholder(project_root, PRESETS[preset])

    if create_gitignore:
        gi_path = os.path.join(project_root, ".gitignore")
        if not os.path.exists(gi_path):
            with open(gi_path, "w") as f:
                f.write(GITIGNORE_TEMPLATE)

    if create_readme:
        rm_path = os.path.join(project_root, "README.md")
        if not os.path.exists(rm_path):
            with open(rm_path, "w") as f:
                f.write(README_TEMPLATE.format(name=name))

    if verbose:
        print(f"Project '{name}' scaffolded at: {project_root}")
        print(f"  Created: {len(created)} folders")
        if skipped:
            print(f"  Skipped (already existed): {len(skipped)} folders")

    return project_root


def _cleanup_and_placeholder(project_root, rel_paths):
    """Add .gitkeep to empty leaf folders so git tracks them; remove stale
    .gitkeep files from folders that now contain real content."""
    for rel_path in rel_paths:
        full_path = os.path.join(project_root, rel_path)
        entries = [e for e in os.listdir(full_path) if e != ".gitkeep"]
        gitkeep_path = os.path.join(full_path, ".gitkeep")
        if not entries:
            if not os.path.exists(gitkeep_path):
                open(gitkeep_path, "w").close()
        else:
            if os.path.exists(gitkeep_path):
                os.remove(gitkeep_path)


def _cli():
    parser = argparse.ArgumentParser(description="Scaffold a project directory tree.")
    parser.add_argument("--name", required=True, help="Project name / folder to create")
    parser.add_argument("--preset", default="ml-project", choices=list(PRESETS.keys()))
    parser.add_argument("--base-dir", default=".", help="Parent directory for the project")
    parser.add_argument("--no-gitignore", action="store_true")
    parser.add_argument("--no-readme", action="store_true")
    args = parser.parse_args()

    scaffold_project(
        name=args.name,
        preset=args.preset,
        base_dir=args.base_dir,
        create_gitignore=not args.no_gitignore,
        create_readme=not args.no_readme,
    )


if __name__ == "__main__":
    _cli()
