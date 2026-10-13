# directory_mapper.py v1.2.2 #

"""Display directory trees and build searchable file tables from notebooks.

Tree functions use only Python's standard library (Python 3.9+). The optional
DataFrame function requires pandas in the calling notebook's environment.
Add the folder containing this file to sys.path, then use::

    import directory_mapper as dm
    dm.map_directory()
    dm.map_directory("~/Documents", depth=2, include_files=True, show_modified=True)

Use help(dm.map_directory) for options or dm.build_tree(...) to get text
without printing it. Importing this module does not scan any directories.
"""

from datetime import datetime
from pathlib import Path
import os
import stat
from typing import Optional, Union

__all__ = ["map_directory", "build_tree", "map_to_dataframe"]
PathInput = Optional[Union[str, os.PathLike]]


def _normalize_extensions(extensions):
    """Normalize one or more suffixes; None means all files."""
    if extensions is None:
        return None
    if isinstance(extensions, str):
        values = [extensions]
    else:
        try:
            values = list(extensions)
        except TypeError as exc:
            raise TypeError("extensions must be a string or iterable of strings.") from exc
    if not values:
        raise ValueError("extensions cannot be empty; use None for all files.")
    normalized = set()
    for value in values:
        if not isinstance(value, str):
            raise TypeError("each extension must be a string.")
        value = value.strip().casefold()
        if not value or value in (".", "*") or "/" in value or "\\" in value:
            raise ValueError(f"Invalid file extension: {value!r}")
        normalized.add(value if value.startswith(".") else "." + value)
    return frozenset(normalized)


def _matches(name, extensions):
    return extensions is None or any(
        len(name) > len(extension) and name.casefold().endswith(extension)
        for extension in extensions
    )


def _local_time(timestamp):
    try:
        return datetime.fromtimestamp(timestamp)
    except (OSError, OverflowError, ValueError, TypeError):
        return None


def _created_time(info):
    # On POSIX, st_ctime is metadata change time, not creation time.
    birth = getattr(info, "st_birthtime", None)
    if birth is None and os.name == "nt":
        birth = info.st_ctime
    return _local_time(birth) if birth is not None else None


def build_tree(
    path: PathInput = None
    , depth: int = 3
    , include_files: bool = False
    , show_modified: bool = False
    , extensions=None
) -> str:
    """Return a directory tree as a multiline string without printing it.

    Parameters
    ----------
    path : str, os.PathLike, or None, default None
        Starting directory. None uses Path.cwd() at call time: the notebook
        kernel's current working directory, which may differ from the folder
        containing the notebook. Accepts absolute or relative paths and '~'.
    depth : int, default 3
        Maximum levels below the starting directory. The root is level 0;
        depth=0 shows only the root, depth=1 includes its immediate children.
        Files and directories both count as entries at their respective level.
        Must be a nonnegative integer (bool is not accepted).
    include_files : bool, default False
        Include files with their complete names and extensions. False lists
        directories only. Directories appear first, followed by files; each
        group is sorted by name without regard to case. Hidden entries are
        included; no project-specific exclusions are applied.
    show_modified : bool, default False
        Append filesystem modification times in local time as yyyy-mm-dd hh:mm.
        A directory's own timestamp describes changes to its directory entries,
        not the latest edit anywhere in its descendants. For symbolic links,
        the timestamp belongs to the link itself.
    extensions : str, iterable of str, or None, default None
        Filter to suffixes such as '.ipynb', 'csv', or ('.csv', '.json').
        Case insensitive; multi-part suffixes such as '.plotly.json' work.
        With a filter, matching files and their ancestor directories appear,
        even if include_files=False. Empty or unrelated folders are hidden.
        The search stops at depth; symlinked directories are not searched.

    Returns
    -------
    str
        Tree text. Directories end in '/'. Symbolic links are labeled and
        never traversed, preventing loops and scans outside the selected tree.
        A symlink explicitly supplied as the root is resolved to its target.
        Unreadable entries/subdirectories are annotated where encountered.
        A directory at the depth limit is not inspected for further contents.

    Raises
    ------
    TypeError
        If depth is not an integer, or a boolean option is not bool.
    ValueError
        If depth is negative.
    FileNotFoundError, NotADirectoryError, PermissionError, OSError
        If the starting directory is missing, invalid, or inaccessible.
        Filesystem changes during scanning may produce inline error markers
        in an unfiltered tree; a filtered scan raises nested errors to avoid
        silently missing matches. This is not a filesystem snapshot.

    Examples
    --------
    >>> tree_text = build_tree(depth=2)
    >>> tree_text = build_tree("~/Documents", include_files=True)
    """
    if isinstance(depth, bool) or not isinstance(depth, int):
        raise TypeError("depth must be a nonnegative integer.")
    if depth < 0:
        raise ValueError("depth must be at least 0.")
    for name, value in (("include_files", include_files), ("show_modified", show_modified)):
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be True or False.")
    selected = _normalize_extensions(extensions)

    root = (Path.cwd() if path is None else Path(path).expanduser()).resolve(strict=True)
    root_stat = root.stat()
    if not stat.S_ISDIR(root_stat.st_mode):
        raise NotADirectoryError(f"Starting path is not a directory: {root}")

    def modified_suffix(info):
        if not show_modified:
            return ""
        try:
            return "  [" + datetime.fromtimestamp(info.st_mtime).strftime("%Y-%m-%d %H:%M") + "]"
        except (OSError, OverflowError, ValueError):
            return "  [modified time unavailable]"

    kept = None

    def children(folder):
        entries = []
        with os.scandir(folder) as scan:
            for entry in scan:
                try:
                    info = entry.stat(follow_symlinks=False)
                    is_link = stat.S_ISLNK(info.st_mode)
                    is_dir = stat.S_ISDIR(info.st_mode)
                    if is_link:
                        is_dir = entry.is_dir(follow_symlinks=True)
                    if not is_dir and not include_files and selected is None:
                        continue
                    if not is_dir and not _matches(entry.name, selected):
                        continue
                    label = entry.name + ("/" if is_dir else "")
                    if is_link:
                        label += " [symlink; not followed]"
                    label += modified_suffix(info)
                    descend = is_dir and not is_link
                except OSError as exc:
                    if selected is not None:
                        raise
                    # Keep errors visible even if the entry's type is unknown.
                    is_dir = False
                    descend = False
                    label = f"{entry.name} [unavailable: {exc.strerror or type(exc).__name__}]"
                child_path = Path(entry.path)
                if kept is not None and child_path not in kept:
                    continue
                entries.append((not is_dir, entry.name.casefold(), entry.name, child_path, label, descend))
        entries.sort(key=lambda item: item[:3])
        return entries

    lines = [str(root) + ("" if str(root).endswith(("/", "\\")) else "/") + modified_suffix(root_stat)]
    if depth == 0:
        return "\n".join(lines)

    if selected is not None:
        # Collect matches and their ancestor folders before drawing the tree.
        matches = set()
        folders = [(root, 0)]
        while folders:
            folder, level = folders.pop()
            for is_file, _, _, child, _, descend in children(folder):
                if descend and level + 1 < depth:
                    folders.append((child, level + 1))
                elif is_file:
                    matches.add(child)
                    parent = child.parent
                    while parent != root:
                        matches.add(parent)
                        parent = parent.parent
        kept = matches
        # children() now returns only paths in kept.

    # Explicit stack supports large requested depths without Python recursion.
    initial = children(root)  # Invalid/unreadable root raises a useful exception.
    stack = [(entry, "", 1, i == len(initial) - 1) for i, entry in reversed(list(enumerate(initial)))]
    while stack:
        entry, prefix, level, last = stack.pop()
        _, _, _, child_path, label, descend = entry
        lines.append(prefix + ("└── " if last else "├── ") + label)
        if not descend or level >= depth:
            continue
        child_prefix = prefix + ("    " if last else "│   ")
        try:
            nested = children(child_path)
        except OSError as exc:
            if selected is not None:
                raise
            lines.append(child_prefix + "└── " + f"[cannot read directory: {exc.strerror or type(exc).__name__}]")
            continue
        stack.extend((item, child_prefix, level + 1, i == len(nested) - 1)
                     for i, item in reversed(list(enumerate(nested))))
    return "\n".join(lines)


def map_directory(
    path: PathInput = None
    , depth: int = 3
    , include_files: bool = False
    , show_modified: bool = False
    , extensions=None
) -> None:
    """Print a directory tree; return None to avoid duplicate notebook output.

    path : str, os.PathLike, or None
        Starting directory; defaults to the kernel's current working directory.
        Supports '~', absolute paths, relative paths, and pathlib.Path objects.
    depth : int, default 3
        Levels below the root (root = 0). Use 0 for the root alone.
    include_files : bool, default False
        Show files with full extensions as well as directories when True.
    show_modified : bool, default False
        Show local filesystem modification time as yyyy-mm-dd hh:mm when True.
        Folder times are NOT the latest modification among all contained files.
    extensions : str, iterable of str, or None, default None
        Search for file suffixes and show matching files plus their ancestor
        folders. For example, extensions='.ipynb' or ('.csv', '.json').

    Hidden entries are included. Symbolic links are labeled but not traversed.
    Inaccessible subdirectories are marked; invalid starting paths raise an
    exception. No files are created or modified by this function.

    Examples
    --------
    >>> import directory_mapper as dm
    >>> dm.map_directory()  # Current directory, 3 levels, directories only.
    >>> dm.map_directory(depth=1)
    >>> dm.map_directory("~/Documents", depth=4, include_files=True)
    >>> dm.map_directory(include_files=True, show_modified=True)

    Use dm.build_tree(...) with the same options when you need a string instead
    of printed output. See help(dm.build_tree) for full behavior and exceptions.
    """
    print(build_tree(path, depth, include_files, show_modified, extensions))


def map_to_dataframe(
    path: PathInput = None
    , depth: int = 3
    , extensions=None
    , verbose_path: bool = True
):
    """Return a pandas DataFrame containing one row per file under a root.

    Parameters
    ----------
    path : str, os.PathLike, or None, default None
        Starting directory; None uses the notebook kernel's current working
        directory. Accepts absolute paths, relative paths, and '~'.
    depth : int, default 3
        Levels below the starting folder (root = 0). A file directly inside
        the root has depth 1. Depth 0 returns an empty table with columns.
    extensions : str, iterable of str, or None, default None
        Optional case-insensitive suffix filter, e.g. '.ipynb', 'csv', or
        ('.csv', '.json'). Any suffix string works. A compound suffix such as
        '.plotly.json' can be searched; extension column uses the *last*
        suffix ('.json'). None returns all files.
    verbose_path : bool, default True
        True: directory_path contains the full absolute folder path.
        False: directory_path begins at the root's folder name, such as
        '/sample_project/results/'. full_path is always absolute.

    Returns
    -------
    pandas.DataFrame
        Columns: name (filename with extension), extension (lowercase final
        suffix, or ''), directory_path (with trailing '/'), parent_directory
        (the containing folder's name), full_path (absolute file path),
        depth, created, modified, size_bytes. Dates are local-time pandas
        datetimes, so they can be sorted and formatted. Size is in bytes.
        On macOS, created uses filesystem birth time; on Windows it uses
        creation time. Where unavailable it is NaT; POSIX metadata change
        time is not mislabeled as creation time. Symbolic links to files use
        the link's metadata. Symlinked directories are not traversed. Hidden
        files are included, and unreadable folders raise an error rather
        than silently producing an incomplete table.

    Raises
    ------
    ImportError
        pandas is missing from the active Python environment.
    TypeError, ValueError, OSError
        Invalid options or an inaccessible directory or entry.

    Examples
    --------
    >>> df = map_to_dataframe('~/Documents', depth=5, extensions='.ipynb')
    >>> df = map_to_dataframe(depth=4, extensions=('.csv', '.json'), verbose_path=False)
    >>> df[['name', 'directory_path', 'modified', 'size_bytes']]
    """
    if isinstance(depth, bool) or not isinstance(depth, int):
        raise TypeError("depth must be a nonnegative integer.")
    if depth < 0:
        raise ValueError("depth must be at least 0.")
    root = (Path.cwd() if path is None else Path(path).expanduser()).resolve(strict=True)
    if not stat.S_ISDIR(root.stat().st_mode):
        raise NotADirectoryError(f"Starting path is not a directory: {root}")
    selected = _normalize_extensions(extensions)
    if not isinstance(verbose_path, bool):
        raise TypeError("verbose_path must be True or False.")
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("map_to_dataframe requires pandas in this Python environment.") from exc

    columns = ["name", "extension", "directory_path", "parent_directory",
               "full_path", "depth", "created", "modified", "size_bytes"]
    rows = []
    if depth:
        stack = [(root, 0)]
        while stack:
            folder, level = stack.pop()
            with os.scandir(folder) as scan:
                entries = sorted(scan, key=lambda item: (item.name.casefold(), item.name))
            for entry in entries:
                info = entry.stat(follow_symlinks=False)
                is_link = stat.S_ISLNK(info.st_mode)
                if stat.S_ISDIR(info.st_mode):
                    if level + 1 < depth:
                        stack.append((Path(entry.path), level + 1))
                    continue
                if is_link and entry.is_dir(follow_symlinks=True):
                    continue
                if not (stat.S_ISREG(info.st_mode) or is_link):
                    continue
                if not _matches(entry.name, selected):
                    continue
                parent = Path(entry.path).parent
                if verbose_path:
                    directory = parent.as_posix().rstrip("/") + "/"
                else:
                    relative = parent.relative_to(root)
                    directory = "/" + root.name + "/" if root.name else "/"
                    if relative.parts:
                        directory += relative.as_posix() + "/"
                rows.append({
                    "name": entry.name,
                    "extension": Path(entry.name).suffix.casefold(),
                    "directory_path": directory,
                    "parent_directory": parent.name,
                    "full_path": str(Path(entry.path)),
                    "depth": level + 1,
                    "created": _created_time(info),
                    "modified": _local_time(info.st_mtime),
                    "size_bytes": info.st_size,
                })

    df = pd.DataFrame.from_records(rows, columns=columns)
    for column in ("created", "modified"):
        df[column] = pd.to_datetime(df[column])
    df["size_bytes"] = pd.array(df["size_bytes"], dtype="Int64")
    return df.sort_values(["directory_path", "name"], key=lambda s: s.str.casefold()
                          if s.dtype == object else s).reset_index(drop=True)
