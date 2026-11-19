#!/usr/bin/env python3
"""Search local file contents from a notebook or terminal (Python 3.10+).

Quick start in Jupyter::

    import string_finder as sf
    sf.help()
    overview = sf.precheck("~/Documents")
    results = sf.search("~/Documents", "family meeting")
    results.show()
    results.to_dataframe()       # pandas is optional
    results.files               # unique paths, including ZIP member paths
    results.issues              # skipped, partial, or unreadable files

Install in the NOTEBOOK's Python environment::

    %pip install tqdm pypdf openpyxl
    # Optional extra formats:
    %pip install xlrd pyxlsb striprtf

Word/PowerPoint OOXML, OpenDocument, notebooks, and text use the standard
library. No Microsoft Office installation is required. Files are read only;
macros, notebook cells, links, and embedded programs are never executed.

An absolute search path works independently of the notebook's working directory.
To import from anywhere, put your helper directory on Python's import path;
see sf.help('setup'). Merely copying a module to a folder does not register it.

This is a text extractor, not OCR or a forensic completeness guarantee. Image
text, embedded attachments, deleted Office revisions, and visual layout are
not searched. See sf.help('formats'), sf.help('limits'), and result.issues.
"""

from __future__ import annotations

import argparse
import codecs
import csv
import importlib
import importlib.util
import io
import json
import math
import os
import re
import sys
import tempfile
import time
import warnings
import zipfile
from collections import Counter
from contextlib import closing, contextmanager
from dataclasses import asdict, dataclass, field
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Callable, Iterable, Iterator
from xml.etree import ElementTree as ET

__version__ = "1.0.0"
__all__ = ["search", "find", "precheck", "help", "supported_formats",
           "SearchReport", "PrecheckReport", "Match", "Issue", "__version__"]

_MIB = 1024 * 1024
_TEXT = frozenset({
    ".txt", ".text", ".csv", ".tsv", ".log", ".md", ".markdown", ".rst",
    ".json", ".jsonl", ".ndjson", ".xml", ".yaml", ".yml", ".toml",
    ".ini", ".cfg", ".conf", ".py", ".pyw", ".r", ".sql", ".sh",
    ".bash", ".zsh", ".js", ".jsx", ".ts", ".tsx", ".css", ".scss",
    ".tex", ".bib", ".srt", ".vtt", ".eml", ".ics", ".vcf", "",
})
_WORD = frozenset({".docx", ".docm", ".dotx", ".dotm"})
_POWERPOINT = frozenset({".pptx", ".pptm", ".ppsx", ".ppsm", ".potx", ".potm"})
_EXCEL = frozenset({".xlsx", ".xlsm", ".xltx", ".xltm"})
_ODF = frozenset({".odt", ".ods", ".odp", ".ott", ".ots", ".otp"})
_HTML = frozenset({".html", ".htm", ".xhtml"})
_OPTIONAL = {".pdf": ("pypdf", "pypdf"), ".xls": ("xlrd", "xlrd"),
             ".xlsb": ("pyxlsb", "pyxlsb"), ".rtf": ("striprtf.striprtf", "striprtf")}
_SUPPORTED = _TEXT | _WORD | _POWERPOINT | _EXCEL | _ODF | _HTML | set(_OPTIONAL) | {".ipynb"}
_LEGACY = {".doc", ".dot", ".ppt", ".pps", ".pot", ".pages", ".numbers", ".key"}


@dataclass(frozen=True)
class Match:
    """One matching text unit; occurrences counts non-overlapping literal hits.

    path uses ``outer.zip!/folder/file.docx`` for archive members. location is
    an extracted-text location, not necessarily a printed page number. snippet
    shows the first occurrence; a unit with several occurrences is one row.
    """

    path: str
    location: str
    occurrences: int
    snippet: str


@dataclass(frozen=True)
class Issue:
    """A coverage gap. status is skipped, error, partial, or warning."""

    path: str
    status: str
    reason: str


@dataclass
class SearchReport:
    """Search results, coverage counts, elapsed time, and export helpers.

    Inspect .issues even when there are no matches. .fully_searched counts
    files whose selected extractor completed without a coverage issue; it
    does NOT mean that images or every possible embedded object were read.
    KeyboardInterrupt returns collected results with .interrupted=True.
    """

    directory: str
    query: str
    matches: list[Match] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)
    attempted_files: int = 0
    fully_searched: int = 0
    partially_searched: int = 0
    failed_files: int = 0
    skipped_files: int = 0
    filtered_files: int = 0
    archives_opened: int = 0
    elapsed_seconds: float = 0.0
    interrupted: bool = False
    options: dict = field(default_factory=dict)

    @property
    def files(self) -> list[str]:
        """Unique matching paths, in discovery order."""
        return list(dict.fromkeys(m.path for m in self.matches))

    @property
    def occurrence_count(self) -> int:
        return sum(m.occurrences for m in self.matches)

    @property
    def summary(self) -> dict:
        """Plain dictionary suitable for logging or notebook inspection."""
        return {
            "matching_files": len(self.files), "matching_units": len(self.matches),
            "occurrences": self.occurrence_count, "attempted_files": self.attempted_files,
            "fully_searched": self.fully_searched, "partially_searched": self.partially_searched,
            "failed_files": self.failed_files, "skipped_files": self.skipped_files,
            "filtered_files": self.filtered_files, "archives_opened": self.archives_opened,
            "issues": len(self.issues), "elapsed_seconds": round(self.elapsed_seconds, 3),
            "interrupted": self.interrupted,
        }

    def __repr__(self) -> str:
        suffix = " INTERRUPTED; partial results." if self.interrupted else ""
        return (f"SearchReport({len(self.files)} matching files, {len(self.matches)} matching units, "
                f"{self.attempted_files} attempted, {len(self.issues)} issues, "
                f"{self.elapsed_seconds:.2f}s).{suffix} "
                "Use .show(), .files, .issues, or .to_dataframe().")

    def show(self, limit: int = 20, *, issue_limit: int = 10) -> None:
        """Print matching excerpts and coverage issues; does not truncate stored results."""
        _nonnegative_int(limit, "limit")
        _nonnegative_int(issue_limit, "issue_limit")
        print(repr(self))
        for match in self.matches[:limit]:
            print(f"\n{match.path}\n  {match.location} | {match.occurrences} occurrence(s)\n  {match.snippet}")
        if len(self.matches) > limit:
            print(f"\n... {len(self.matches) - limit} more matching units stored.")
        _print_issues(self.issues, issue_limit)

    def to_dataframe(self):
        """Return matches as a pandas DataFrame (optional: %pip install pandas)."""
        return _dataframe(self.matches, ["path", "location", "occurrences", "snippet"])

    def issues_to_dataframe(self):
        """Return coverage issues as a pandas DataFrame."""
        return _dataframe(self.issues, ["path", "status", "reason"])

    def to_csv(self, path: str | os.PathLike) -> Path:
        """Export matching units to UTF-8 CSV; only this explicit call writes a file."""
        return _write_csv(path, self.matches, ["path", "location", "occurrences", "snippet"])

    def issues_to_csv(self, path: str | os.PathLike) -> Path:
        """Export coverage issues to UTF-8 CSV."""
        return _write_csv(path, self.issues, ["path", "status", "reason"])


@dataclass
class PrecheckReport:
    """Metadata inventory, not a trial text extraction or an execution-time forecast.

    Files inside ZIPs are counted separately from filesystem files. Large-file
    sizes for members are their declared uncompressed sizes. Nested ZIPs must
    be decompressed to inspect their directories; document bodies are not read.
    A candidate can still fail extraction, need a dependency, or contain images.
    """

    directory: str
    filesystem_files: int = 0
    archive_members: int = 0
    candidate_files: int = 0
    filesystem_bytes: int = 0
    member_bytes: int = 0
    archives_opened: int = 0
    filtered_files: int = 0
    skipped_files: int = 0
    large_file_mb: float = 10
    large_files: list[dict] = field(default_factory=list)
    by_extension: Counter = field(default_factory=Counter)
    issues: list[Issue] = field(default_factory=list)
    dependencies: dict = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    interrupted: bool = False

    def __repr__(self) -> str:
        return (f"PrecheckReport({self.filesystem_files} filesystem files, "
                f"{self.archive_members} ZIP members, {self.candidate_files} candidates, "
                f"{len(self.large_files)} files > {self.large_file_mb:g} MiB, "
                f"{len(self.issues)} issues" + (", INTERRUPTED)" if self.interrupted else ")"))

    def show(self, limit: int = 10) -> None:
        """Print counts, extension mix, largest files, dependencies, and issues."""
        _nonnegative_int(limit, "limit")
        print(repr(self))
        print(f"Filesystem size: {self.filesystem_bytes / _MIB:,.2f} MiB")
        print(f"ZIP member sizes (declared; nested layers overlap): {self.member_bytes / _MIB:,.2f} MiB")
        print(f"Opened archives: {self.archives_opened}; filtered: {self.filtered_files}; skipped: {self.skipped_files}")
        print("Extension mix: " + ", ".join(f"{ext or '[no extension]'}: {n}" for ext, n in self.by_extension.most_common(limit)))
        for row in sorted(self.large_files, key=lambda r: r["size_bytes"], reverse=True)[:limit]:
            print(f"  {row['size_bytes'] / _MIB:,.2f} MiB  {row['path']}")
        print("Dependencies: " + "; ".join(f"{k}={'available' if v else 'not installed'}" for k, v in self.dependencies.items()))
        if self.interrupted:
            print("Inventory interrupted; these are partial counts.")
        _print_issues(self.issues, limit)


def _print_issues(issues: list[Issue], limit: int) -> None:
    if issues:
        print(f"\nCoverage issues ({len(issues)}; inspect .issues for the full list):")
        for issue in issues[:limit]:
            print(f"  [{issue.status}] {issue.path}: {issue.reason}")
        if len(issues) > limit:
            print(f"  ... {len(issues) - limit} more.")


def _dataframe(rows, columns):
    pd = _require("pandas", "pandas")
    return pd.DataFrame([asdict(row) for row in rows], columns=columns)


def _write_csv(path, rows, columns) -> Path:
    target = Path(path).expanduser().absolute()
    with target.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    return target


def _nonnegative_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _positive(value, name, *, allow_none=True):
    if value is None and allow_none:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number" + (" or None" if allow_none else ""))


def _extensions(values):
    if values is None:
        return None
    if isinstance(values, str):
        values = values.split(",")
    result = set()
    for value in values:
        if not isinstance(value, str):
            raise TypeError("Extensions must be strings, e.g. ['.docx', '.pdf']")
        value = value.strip().lower()
        if value.startswith("*."):
            value = value[1:]
        if value and not value.startswith("."):
            value = "." + value
        result.add(value)
    return frozenset(result)


@dataclass
class _Options:
    recursive: bool = True
    include_extensions: Iterable[str] | str | None = None
    exclude_extensions: Iterable[str] | str | None = None
    search_zips: bool = True
    include_hidden: bool = False
    max_file_mb: float | None = 250
    max_zip_depth: int = 3
    max_archive_member_mb: float | None = 250
    max_archive_total_mb: float | None = 2048
    max_archive_ratio: float | None = 1000
    max_archive_members: int = 100_000
    encoding: str | None = None

    def __post_init__(self):
        self.include_extensions = _extensions(self.include_extensions)
        self.exclude_extensions = _extensions(self.exclude_extensions) or frozenset()
        for name in ("recursive", "search_zips", "include_hidden"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be True or False")
        _nonnegative_int(self.max_zip_depth, "max_zip_depth")
        _nonnegative_int(self.max_archive_members, "max_archive_members")
        for name in ("max_file_mb", "max_archive_member_mb", "max_archive_total_mb", "max_archive_ratio"):
            _positive(getattr(self, name), name)
        if self.encoding is not None:
            codecs.lookup(self.encoding)


class _Skip(Exception):
    pass


def _require(module: str, package: str):
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise _Skip(f"Missing optional dependency: install in this notebook with %pip install {package}") from exc


def _available(module):
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError, AttributeError):
        return False


@dataclass
class _Source:
    path: str
    name: str
    size: int
    opener: Callable
    in_archive: bool = False

    @property
    def ext(self):
        return PurePosixPath(self.name).suffix.lower()


@contextmanager
def _member_stream(archive, info):
    # No ZIP member is extracted into the user's folders. Larger members spool
    # to an automatically removed temporary file so PDF/Excel readers can seek.
    with archive.open(info) as raw, tempfile.SpooledTemporaryFile(max_size=8 * _MIB, mode="w+b") as spool:
        copied = 0
        while chunk := raw.read(1024 * 1024):
            copied += len(chunk)
            if copied > info.file_size:
                raise _Skip("ZIP member expanded beyond its declared size")
            spool.write(chunk)
        spool.seek(0)
        yield spool


class _Walker:
    def __init__(self, root, options, report, bar):
        self.root, self.options, self.report, self.bar = root, options, report, bar
        self.archive_bytes = 0
        self.archive_entries = 0

    def issue(self, path, status, reason):
        self.report.issues.append(Issue(str(path), status, str(reason)))
        if status in {"skipped", "error"}:
            self.report.skipped_files += 1

    def __iter__(self):
        if self.root.is_file():
            yield from self._file(self.root)
        else:
            yield from self._directory(self.root)

    def _directory(self, directory):
        try:
            with os.scandir(directory) as entries:
                # Keep a directory at a time in memory; don't build a whole-tree list.
                ordered = sorted(entries, key=lambda e: e.name.casefold())
            for entry in ordered:
                if not self.options.include_hidden and entry.name.startswith("."):
                    self.report.filtered_files += 1
                    continue
                try:
                    if entry.is_symlink():
                        self.issue(entry.path, "skipped", "Symbolic links are not followed")
                    elif entry.is_dir(follow_symlinks=False):
                        if self.options.recursive:
                            yield from self._directory(Path(entry.path))
                    elif entry.is_file(follow_symlinks=False):
                        yield from self._file(Path(entry.path))
                except OSError as exc:
                    self.issue(entry.path, "error", f"Filesystem access failed: {exc}")
        except OSError as exc:
            self.issue(directory, "error", f"Cannot list directory: {exc}")

    def _file(self, path):
        try:
            size = path.stat().st_size
        except OSError as exc:
            self.issue(path, "error", f"Cannot stat file: {exc}")
            return
        source = _Source(str(path), path.name, size, lambda: path.open("rb"))
        yield from self._visit(source, 0)

    def _visit(self, source, depth):
        opt, rep = self.options, self.report
        if self.bar is not None:
            self.bar.update(1)
        if not source.in_archive:
            self._record_inventory(source)
        if source.name.startswith("~$") or source.name.startswith("._"):
            rep.filtered_files += 1
            return
        if source.ext in opt.exclude_extensions:
            rep.filtered_files += 1
            return
        is_zip = source.ext == ".zip"
        # Include filters select leaf documents; ZIPs remain traversable containers.
        if not is_zip and opt.include_extensions is not None and source.ext not in opt.include_extensions:
            rep.filtered_files += 1
            return
        if opt.max_file_mb is not None and source.size > opt.max_file_mb * _MIB:
            self.issue(source.path, "skipped", f"File exceeds max_file_mb={opt.max_file_mb:g}")
            return
        if is_zip:
            if not opt.search_zips:
                rep.filtered_files += 1
                return
            if depth >= opt.max_zip_depth:
                self.issue(source.path, "skipped", f"ZIP nesting limit reached (max_zip_depth={opt.max_zip_depth})")
                return
            yield from self._archive(source, depth)
        elif source.ext not in _SUPPORTED:
            advice = " Convert to .docx/.xlsx/.pptx or plain text first." if source.ext in _LEGACY else ""
            self.issue(source.path, "skipped", f"Unsupported extension: {source.ext or '[none]'}.{advice}")
        else:
            if isinstance(rep, PrecheckReport):
                rep.candidate_files += 1
            yield source

    def _record_inventory(self, source):
        rep = self.report
        if not isinstance(rep, PrecheckReport):
            return
        if source.in_archive:
            rep.archive_members += 1
            rep.member_bytes += source.size
        else:
            rep.filesystem_files += 1
            rep.filesystem_bytes += source.size
        rep.by_extension[source.ext] += 1
        if source.size > rep.large_file_mb * _MIB:
            rep.large_files.append({"path": source.path, "size_bytes": source.size,
                                    "in_archive": source.in_archive})

    def _archive(self, source, depth):
        opt = self.options
        try:
            with source.opener() as handle, zipfile.ZipFile(handle) as archive:
                self.report.archives_opened += 1
                for info in archive.infolist():
                    if info.is_dir():
                        continue
                    self.archive_entries += 1
                    if self.archive_entries > opt.max_archive_members:
                        self.issue(source.path, "skipped", "Global max_archive_members limit reached; remaining entries not inspected")
                        break
                    # Treat both separators defensively; never extract these paths.
                    parts = PurePosixPath(info.filename.replace("\\", "/")).parts
                    if not opt.include_hidden and any(p.startswith(".") or p == "__MACOSX" for p in parts):
                        self.report.filtered_files += 1
                        continue
                    member = _Source(f"{source.path}!/{info.filename}", parts[-1] if parts else info.filename,
                                     info.file_size, lambda i=info: _member_stream(archive, i), True)
                    self._record_inventory(member)
                    # Don't reserve decompression budgets for excluded/unsupported leaves.
                    if member.ext in opt.exclude_extensions:
                        self.report.filtered_files += 1
                        continue
                    if member.ext != ".zip" and opt.include_extensions is not None and member.ext not in opt.include_extensions:
                        self.report.filtered_files += 1
                        continue
                    if info.flag_bits & 1:
                        self.issue(member.path, "skipped", "Encrypted ZIP member; passwords are not requested")
                        continue
                    if opt.max_archive_member_mb is not None and info.file_size > opt.max_archive_member_mb * _MIB:
                        self.issue(member.path, "skipped", "ZIP member exceeds max_archive_member_mb")
                        continue
                    if opt.max_archive_ratio is not None and info.file_size / max(1, info.compress_size) > opt.max_archive_ratio:
                        self.issue(member.path, "skipped", "ZIP member exceeds max_archive_ratio")
                        continue
                    if member.ext in _SUPPORTED or member.ext == ".zip":
                        if opt.max_archive_total_mb is not None and self.archive_bytes + info.file_size > opt.max_archive_total_mb * _MIB:
                            self.issue(member.path, "skipped", "Global max_archive_total_mb budget exceeded")
                            continue
                        self.archive_bytes += info.file_size
                    yield from self._visit(member, depth + 1)
        except Exception as exc:
            self.issue(source.path, "error", f"Cannot read ZIP: {type(exc).__name__}: {exc}")


def _root(directory) -> Path:
    path = Path(directory).expanduser().absolute()
    if path.is_symlink():
        raise ValueError("Choose the actual file/directory rather than a symbolic link")
    if not path.exists():
        raise FileNotFoundError(path)
    if not path.is_dir() and not path.is_file():
        raise ValueError(f"Not a regular file or directory: {path}")
    return path


def _progress(enabled, label):
    if not isinstance(enabled, bool):
        raise TypeError("progress must be True or False")
    if not enabled:
        return None
    try:
        from tqdm.auto import tqdm
        return tqdm(desc=label, unit="file", dynamic_ncols=True)
    except ImportError:
        warnings.warn("tqdm is not installed; continuing without a progress bar. In Jupyter: %pip install tqdm", stacklevel=3)
        return None


def precheck(directory: str | os.PathLike, *, large_file_mb: float = 10,
             progress: bool = True, **options) -> PrecheckReport:
    """Inventory a directory (or single file) without extracting document text.

    Defaults: recurse, inspect ZIPs, omit hidden paths and symlinks, count files
    larger than 10 MiB. ZIP recursion is independent of filesystem recursion.
    Accepted options are the traversal/filter/size/encoding options in search(),
    excluding case_sensitive, normalize_whitespace, context_chars, and match caps.

    Examples::

        sf.precheck('~/Documents').show()
        sf.precheck('~/Documents', recursive=False, search_zips=False).show()
        sf.precheck('~/Documents', include_extensions=['docx', 'pdf']).show()

    The progress counter grows as files are discovered; no extra full-tree scan
    is needed. Metadata can change between this inventory and a later search.
    """
    root = _root(directory)
    _positive(large_file_mb, "large_file_mb", allow_none=False)
    opt = _Options(**options)
    report = PrecheckReport(str(root), large_file_mb=large_file_mb)
    report.dependencies = {name: _available(name) for name in ("tqdm", "pypdf", "openpyxl", "xlrd", "pyxlsb", "striprtf")}
    started = time.perf_counter()
    bar = _progress(progress, "Inventory")
    walker = iter(_Walker(root, opt, report, bar))
    try:
        for _ in walker:
            pass
    except KeyboardInterrupt:
        report.interrupted = True
    finally:
        walker.close()
        if bar is not None:
            bar.close()
        report.elapsed_seconds = time.perf_counter() - started
    return report


def search(
    directory: str | os.PathLike,
    query: str,
    *,
    recursive: bool = True,
    case_sensitive: bool = False,
    include_extensions: Iterable[str] | str | None = None,
    exclude_extensions: Iterable[str] | str | None = None,
    search_zips: bool = True,
    progress: bool = True,
    normalize_whitespace: bool = True,
    max_matches_per_file: int | None = None,
    context_chars: int = 90,
    include_hidden: bool = False,
    max_file_mb: float | None = 250,
    max_zip_depth: int = 3,
    max_archive_member_mb: float | None = 250,
    max_archive_total_mb: float | None = 2048,
    max_archive_ratio: float | None = 1000,
    max_archive_members: int = 100_000,
    encoding: str | None = None,
) -> SearchReport:
    """Find a literal string/phrase inside local files and ZIP members.

    Parameters
    ----------
    directory : path-like
        Folder to search, or a single file. '~' expands to your home directory.
        Use an absolute path for independence from the notebook working folder.
    query : str
        Nonempty literal text; punctuation such as '.', '+', and '[' is literal.
        Case-insensitive matching uses Unicode-aware regular-expression rules.
    recursive : bool, default True
        Descend into filesystem subfolders. False still searches ZIP contents.
    case_sensitive : bool, default False
        True requires matching letter case; extension filters remain insensitive.
    include_extensions, exclude_extensions : iterable[str] or comma-separated str
        Examples: ['.docx', '.pdf'], 'xlsx,csv'. Exclusion wins. Include filters
        select documents INSIDE ZIPs too; no need to include '.zip' to traverse.
        Explicitly excluding '.zip' disables traversal. None includes all known
        formats; [] includes no leaf documents. Use '' for extensionless text.
    search_zips : bool, default True
        Open ZIP containers, including nested ZIPs up to max_zip_depth.
    progress : bool, default True
        Use tqdm.auto for notebooks/terminals. The counter grows with discovery;
        it has no percentage/ETA because there is no preliminary full-tree scan.
    normalize_whitespace : bool, default True
        Treat whitespace runs as one space in query and extracted units. Helps
        with PDF line wrapping. Search units: text line, Word paragraph, Excel
        cell, PowerPoint paragraph, PDF page, notebook cell. Matches do not span
        units (e.g. pages, cells, paragraphs, or separate plain-text lines).
    max_matches_per_file : int or None, default None
        Cap matching UNITS per file, not occurrences; None keeps all. Reaching
        a cap stops that file and is recorded as a coverage issue.
    context_chars : int, default 90
        Characters shown before and after the first occurrence in each unit.
    include_hidden : bool, default False
        Include dot-prefixed paths and __MACOSX ZIP entries. Office '~$' lock
        files and Apple '._' metadata files are always omitted. Symlinks are
        never followed. An explicitly selected hidden root is still searched.
    max_file_mb : float or None, default 250
        Skip larger filesystem files or archive leaves; sizes are in MiB.
    max_zip_depth : int, default 3
        ZIP layers; 1 searches direct members, 0 skips all ZIP containers.
    max_archive_member_mb : float or None, default 250
        Maximum uncompressed ZIP/Office-package member size.
    max_archive_total_mb : float or None, default 2048
        Global declared bytes for selected ZIP members, including nested ZIP
        layers. Also bounds each Office package's declared uncompressed size.
    max_archive_ratio : float or None, default 1000
        Skip highly compressed ZIP/Office members above this ratio.
    max_archive_members : int, default 100000
        Global ordinary-ZIP entry inspection cap and per-Office package cap.
    encoding : str or None
        Override text decoding. Otherwise detect UTF BOMs/UTF-8, then use
        Windows-1252. Fallback decoding is reported because it can be ambiguous.

    Returns
    -------
    SearchReport
        .matches, .files, .issues, .summary, .show(), .to_dataframe(), .to_csv().
        Failed extraction never gets labeled a successful no-match. Interrupting
        returns partial results. Text extraction itself has format limitations;
        read sf.help('formats') and sf.help('limits').
    """
    root = _root(directory)
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a nonempty, non-whitespace string")
    for value, name in ((case_sensitive, "case_sensitive"), (normalize_whitespace, "normalize_whitespace")):
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be True or False")
    _nonnegative_int(context_chars, "context_chars")
    if max_matches_per_file is not None:
        _nonnegative_int(max_matches_per_file, "max_matches_per_file")
        if max_matches_per_file == 0:
            raise ValueError("max_matches_per_file must be >= 1 or None")
    opt = _Options(recursive, include_extensions, exclude_extensions, search_zips,
                   include_hidden, max_file_mb, max_zip_depth, max_archive_member_mb,
                   max_archive_total_mb, max_archive_ratio, max_archive_members, encoding)
    needle = _normalize(query) if normalize_whitespace else query
    pattern = re.compile(re.escape(needle), 0 if case_sensitive else re.IGNORECASE)
    report = SearchReport(str(root), query)
    report.options = asdict(opt)
    report.options.update(case_sensitive=case_sensitive, normalize_whitespace=normalize_whitespace,
                          max_matches_per_file=max_matches_per_file, context_chars=context_chars)
    started = time.perf_counter()
    bar = _progress(progress, "Searching")
    walker = iter(_Walker(root, opt, report, bar))
    current_source = None
    try:
        for source in walker:
            current_source = source
            report.attempted_files += 1
            before_issues, before_matches = len(report.issues), len(report.matches)
            local_matches = 0
            try:
                with source.opener() as handle, closing(_units(handle, source, opt, report.issues)) as units:
                    for location, text in units:
                        text = _normalize(text) if normalize_whitespace else text
                        first = pattern.search(text)
                        if first is None:
                            continue
                        count = sum(1 for _ in pattern.finditer(text))
                        start = max(0, first.start() - context_chars)
                        end = min(len(text), first.end() + context_chars)
                        snippet = ("…" if start else "") + text[start:end] + ("…" if end < len(text) else "")
                        report.matches.append(Match(source.path, location, count, snippet))
                        local_matches += 1
                        if max_matches_per_file is not None and local_matches >= max_matches_per_file:
                            report.issues.append(Issue(source.path, "partial", "max_matches_per_file reached; remaining text not searched"))
                            break
            except _Skip as exc:
                report.issues.append(Issue(source.path, "skipped", str(exc)))
            except Exception as exc:
                report.issues.append(Issue(source.path, "error", f"{type(exc).__name__}: {exc}"))
            file_issues = report.issues[before_issues:]
            if not file_issues:
                report.fully_searched += 1
            elif len(report.matches) > before_matches or any(i.status in {"warning", "partial"} for i in file_issues):
                report.partially_searched += 1
            else:
                report.failed_files += 1
            current_source = None
    except KeyboardInterrupt:
        report.interrupted = True
        if current_source is not None:
            report.partially_searched += 1
            report.issues.append(Issue(current_source.path, "partial", "Search interrupted while reading this file"))
    finally:
        walker.close()
        if bar is not None:
            bar.close()
        report.elapsed_seconds = time.perf_counter() - started
    return report


find = search


def _normalize(text):
    return " ".join(text.split())


def _encoding(handle, override, issues, path):
    sample = handle.read(65536)
    at_eof = handle.read(1) == b""
    handle.seek(0)
    if override:
        return override
    for bom, name in ((codecs.BOM_UTF32_LE, "utf-32"), (codecs.BOM_UTF32_BE, "utf-32"),
                      (codecs.BOM_UTF8, "utf-8-sig"), (codecs.BOM_UTF16_LE, "utf-16"),
                      (codecs.BOM_UTF16_BE, "utf-16")):
        if sample.startswith(bom):
            return name
    if b"\0" in sample:
        raise _Skip("Binary-looking content or UTF-16 without BOM; set encoding='utf-16-le'/'utf-16-be' if appropriate")
    try:
        # An incomplete multibyte sequence is allowed only at a sample boundary.
        codecs.getincrementaldecoder("utf-8")().decode(sample, final=at_eof)
    except UnicodeDecodeError:
        issues.append(Issue(path, "warning", "Text decoded as Windows-1252 fallback; set encoding= explicitly if needed"))
        return "cp1252"
    return "utf-8-sig"


@contextmanager
def _text_handle(handle, opt, issues, path):
    encoding = _encoding(handle, opt.encoding, issues, path)
    wrapper = io.TextIOWrapper(handle, encoding=encoding, errors="strict", newline=None)
    try:
        yield wrapper
    finally:
        # The caller owns the binary stream. Keep it open until its own context exits.
        wrapper.detach()


def _guard_package(handle, opt):
    handle.seek(0)
    with zipfile.ZipFile(handle) as archive:
        entries = archive.infolist()
        if len(entries) > opt.max_archive_members:
            raise _Skip("Office/OpenDocument package exceeds max_archive_members")
        total = 0
        for info in entries:
            total += info.file_size
            if info.flag_bits & 1:
                raise _Skip("Encrypted Office/OpenDocument package")
            if opt.max_archive_member_mb is not None and info.file_size > opt.max_archive_member_mb * _MIB:
                raise _Skip("Office/OpenDocument component exceeds max_archive_member_mb")
            if opt.max_archive_ratio is not None and info.file_size / max(1, info.compress_size) > opt.max_archive_ratio:
                raise _Skip("Office/OpenDocument component exceeds max_archive_ratio")
        if opt.max_archive_total_mb is not None and total > opt.max_archive_total_mb * _MIB:
            raise _Skip("Office/OpenDocument package exceeds max_archive_total_mb")
    handle.seek(0)


def _xml(archive, name):
    raw = archive.read(name)
    # Office XML has no need for DTD/entity declarations. Reject rather than
    # expanding entities, including documents using an unusual XML encoding.
    compact = raw.replace(b"\0", b"").upper()
    if b"<!DOCTYPE" in compact or b"<!ENTITY" in compact:
        raise _Skip(f"XML DTD/entity declarations are not supported: {name}")
    return ET.fromstring(raw)


def _paragraph_text(paragraph, namespace):
    pieces = []
    def collect(node):
        if node is not paragraph and node.tag == namespace + "p":
            return
        if node.tag == namespace + "t":
            pieces.append(node.text or "")
        elif node.tag == namespace + "tab":
            pieces.append("\t")
        elif node.tag in {namespace + "br", namespace + "cr"}:
            pieces.append("\n")
        for child in node:
            collect(child)
    collect(paragraph)
    return "".join(pieces)


def _word_units(handle, opt, issues, path):
    with zipfile.ZipFile(handle) as archive:
        names = archive.namelist()
        main = ["word/document.xml"] if "word/document.xml" in names else []
        if not main:
            raise _Skip("Word package has no word/document.xml")
        extra = sorted(n for n in names if re.fullmatch(r"word/(?:header\d+|footer\d+|footnotes|endnotes|comments)\.xml", n))
        for name in main + extra:
            root = _xml(archive, name)
            namespace = root.tag.split("}")[0] + "}"
            if (next(root.iter(namespace + "del"), None) is not None
                    or next(root.iter(namespace + "moveFrom"), None) is not None):
                issues.append(Issue(path, "partial", f"Deleted/moved-from revisions omitted in {name}"))
            # Remove deleted subtrees, then extract every current paragraph,
            # including nested textboxes without folding them into outer text.
            for parent in list(root.iter()):
                for child in list(parent):
                    if child.tag in {namespace + "del", namespace + "moveFrom"}:
                        parent.remove(child)
            for index, paragraph in enumerate(root.iter(namespace + "p"), 1):
                text = _paragraph_text(paragraph, namespace)
                if text:
                    yield f"{name}: paragraph {index}", text


def _ppt_units(handle, opt, issues, path):
    with zipfile.ZipFile(handle) as archive:
        names = set(archive.namelist())
        if "ppt/presentation.xml" not in names:
            raise _Skip("PowerPoint package has no ppt/presentation.xml")
        # Slide order comes from relationships, not slide filename numbering.
        rels = _xml(archive, "ppt/_rels/presentation.xml.rels")
        targets = {r.attrib["Id"]: r.attrib.get("Target", "") for r in rels
                   if r.attrib.get("TargetMode") != "External"}
        presentation = _xml(archive, "ppt/presentation.xml")
        slide_ids = [next((v for k, v in e.attrib.items() if k.endswith("}id")), None) for e in presentation.iter()
                     if e.tag.rsplit("}", 1)[-1] == "sldId"]
        for number, relationship in enumerate(slide_ids, 1):
            target = targets.get(relationship, "")
            part = _part_path("ppt/presentation.xml", target)
            if part not in names:
                raise _Skip(f"Missing slide component: {part}")
            root = _xml(archive, part)
            for index, paragraph in enumerate(_drawing_paragraphs(root), 1):
                text = _paragraph_text(paragraph, paragraph.tag.split("}")[0] + "}")
                if text:
                    yield f"slide {number}: paragraph {index}", text
            rel_part = str(PurePosixPath(part).parent / "_rels" / (PurePosixPath(part).name + ".rels"))
            if rel_part in names:
                for rel in _xml(archive, rel_part):
                    if rel.attrib.get("Type", "").endswith("/notesSlide") and rel.attrib.get("TargetMode") != "External":
                        note_part = _part_path(part, rel.attrib["Target"])
                        for index, paragraph in enumerate(_drawing_paragraphs(_xml(archive, note_part)), 1):
                            text = _paragraph_text(paragraph, paragraph.tag.split("}")[0] + "}")
                            if text:
                                yield f"slide {number}: notes paragraph {index}", text
        # Chart cached strings and comments are useful meeting-agenda content.
        for part in sorted(names):
            if re.fullmatch(r"ppt/(?:comments|charts)/[^/]+\.xml", part):
                root = _xml(archive, part)
                for index, node in enumerate(root.iter(), 1):
                    if node.tag.rsplit("}", 1)[-1] in {"t", "text", "v"} and node.text:
                        yield f"{part}: text {index}", node.text


def _part_path(base, target):
    import posixpath
    if target.startswith("/"):
        return target.lstrip("/")
    return posixpath.normpath(posixpath.join(posixpath.dirname(base), target))


def _drawing_paragraphs(root):
    # Both Transitional and Strict OOXML namespaces occur in saved presentations.
    return (node for node in root.iter() if node.tag.endswith("}p")
            and "drawingml" in node.tag.lower())


def _excel_units(handle, opt, issues, path):
    xl = _require("openpyxl", "openpyxl")
    # Separate read-only views: formula source and previously saved cached value.
    # Never calculate formulas or fetch external links.
    formulas = xl.load_workbook(handle, read_only=True, data_only=False, keep_links=False)
    cached = None
    try:
        handle.seek(0)
        cached = xl.load_workbook(handle, read_only=True, data_only=True, keep_links=False)
        missing_cache = False
        for sheet, values in zip(formulas.worksheets, cached.worksheets):
            sheet.reset_dimensions()
            values.reset_dimensions()
            yield f"sheet {sheet.title!r}: name", sheet.title
            for row, value_row in zip(sheet.iter_rows(), values.iter_rows()):
                for cell, saved in zip(row, value_row):
                    if cell.value is None:
                        continue
                    location = f"sheet {sheet.title!r}: {cell.coordinate}"
                    raw_value = getattr(cell.value, "text", cell.value) if cell.data_type == "f" else cell.value
                    yield location + (" [formula]" if cell.data_type == "f" else ""), str(raw_value)
                    if cell.data_type == "f":
                        if saved.value is None:
                            missing_cache = True
                        elif saved.value != cell.value:
                            yield location + " [cached result]", str(saved.value)
        if missing_cache:
            issues.append(Issue(path, "partial", "Some Excel formulas have no cached result; formula source searched, results not calculated"))
    finally:
        if cached is not None:
            cached.close()
        formulas.close()
    handle.seek(0)
    with zipfile.ZipFile(handle) as archive:
        for part in sorted(archive.namelist()):
            if re.fullmatch(r"xl/(?:comments/comment\d+|comments\d+)\.xml", part):
                root = _xml(archive, part)
                for node in root.iter():
                    if node.tag.rsplit("}", 1)[-1] == "comment":
                        text = "".join(e.text or "" for e in node.iter() if e.tag.rsplit("}", 1)[-1] == "t")
                        yield f"{part}: comment {node.attrib.get('ref', '')}", text


def _odf_units(handle):
    ns = "{urn:oasis:names:tc:opendocument:xmlns:text:1.0}"
    table_ns = "{urn:oasis:names:tc:opendocument:xmlns:table:1.0}"
    with zipfile.ZipFile(handle) as archive:
        root = _xml(archive, "content.xml")
        if any(e.tag.rsplit("}", 1)[-1] == "encryption-data" for e in _xml(archive, "META-INF/manifest.xml").iter()):
            raise _Skip("Encrypted OpenDocument file")
        for index, paragraph in enumerate(root.iter(), 1):
            if paragraph.tag in {ns + "p", ns + "h"}:
                def text_content(node):
                    value = node.text or ""
                    for child in node:
                        if child.tag == ns + "s":
                            count = min(int(child.attrib.get(ns + "c", 1)), 1_000_000)
                            value += " " * max(0, count)
                        elif child.tag == ns + "tab":
                            value += "\t"
                        elif child.tag == ns + "line-break":
                            value += "\n"
                        else:
                            value += text_content(child)
                        value += child.tail or ""
                    return value
                yield f"content.xml: paragraph {index}", text_content(paragraph)
            elif paragraph.tag == table_ns + "table-cell":
                # Numeric/date/formula attributes may not have visible paragraphs.
                for key, value in paragraph.attrib.items():
                    if key.rsplit("}", 1)[-1] in {"value", "date-value", "string-value", "formula"}:
                        yield f"content.xml: cell element {index}", value


class _VisibleHTML(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.hidden = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.hidden += 1
        elif tag in {"p", "div", "br", "li", "tr", "h1", "h2", "h3"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.hidden = max(0, self.hidden - 1)
        elif tag in {"p", "div", "li", "tr"}:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def _units(handle, source, opt, issues):
    ext, path = source.ext, source.path
    if ext in _WORD | _POWERPOINT | _EXCEL | _ODF | {".xlsb"}:
        _guard_package(handle, opt)
    if ext in _WORD:
        yield from _word_units(handle, opt, issues, path)
    elif ext in _POWERPOINT:
        yield from _ppt_units(handle, opt, issues, path)
    elif ext in _EXCEL:
        yield from _excel_units(handle, opt, issues, path)
    elif ext in _ODF:
        yield from _odf_units(handle)
    elif ext == ".pdf":
        pdf = _require("pypdf", "pypdf")
        reader = pdf.PdfReader(handle, strict=False)
        if reader.is_encrypted and not reader.decrypt(""):
            raise _Skip("Password-protected PDF; passwords are not requested")
        for number, page in enumerate(reader.pages, 1):
            try:
                text = page.extract_text() or ""
                if text.strip():
                    yield f"page {number}", text
                else:
                    issues.append(Issue(path, "partial", f"Page {number} has no extractable text; blank, scanned, or image-based (OCR not included)"))
            except Exception as exc:
                issues.append(Issue(path, "partial", f"Page {number} extraction failed: {type(exc).__name__}: {exc}"))
    elif ext == ".xls":
        xlrd = _require("xlrd", "xlrd")
        book = xlrd.open_workbook(file_contents=handle.read(), on_demand=True)
        try:
            for sheet in book.sheets():
                yield f"sheet {sheet.name!r}: name", sheet.name
                for row in range(sheet.nrows):
                    for col in range(sheet.ncols):
                        value = sheet.cell_value(row, col)
                        if value != "":
                            yield f"sheet {sheet.name!r}: row {row + 1}, column {col + 1}", str(value)
        finally:
            book.release_resources()
    elif ext == ".xlsb":
        xlsb = _require("pyxlsb", "pyxlsb")
        with xlsb.open_workbook(handle) as book:
            for name in book.sheets:
                yield f"sheet {name!r}: name", name
                with book.get_sheet(name) as sheet:
                    for row in sheet.rows():
                        for cell in row:
                            if cell.v is not None:
                                yield f"sheet {name!r}: row {cell.r + 1}, column {cell.c + 1}", str(cell.v)
    elif ext == ".ipynb":
        with _text_handle(handle, opt, issues, path) as text:
            notebook = json.load(text)
        for number, cell in enumerate(notebook.get("cells", []), 1):
            source_text = cell.get("source", "")
            yield f"notebook cell {number} ({cell.get('cell_type', '?')}): source", "".join(source_text)
            for index, output in enumerate(cell.get("outputs", []), 1):
                value = output.get("text", output.get("data", {}).get("text/plain", ""))
                if value:
                    yield f"notebook cell {number}: output {index}", "".join(value)
    elif ext == ".rtf":
        rtf = _require("striprtf.striprtf", "striprtf")
        with _text_handle(handle, opt, issues, path) as text:
            value = rtf.rtf_to_text(text.read())
        for number, line in enumerate(value.splitlines(), 1):
            yield f"RTF text line {number}", line
    elif ext in _HTML:
        with _text_handle(handle, opt, issues, path) as text:
            parser = _VisibleHTML()
            for block in iter(lambda: text.read(65536), ""):
                parser.feed(block)
            parser.close()
        for number, line in enumerate("".join(parser.parts).splitlines(), 1):
            yield f"HTML text line {number}", line
    else:
        with _text_handle(handle, opt, issues, path) as text:
            for number, line in enumerate(text, 1):
                yield f"line {number}", line.rstrip("\r\n")


def supported_formats() -> dict[str, str]:
    """Return extension groups and requirements; does not install dependencies."""
    return {
        "Word": ", ".join(sorted(_WORD)) + " (standard library)",
        "Excel": ", ".join(sorted(_EXCEL)) + " (openpyxl); .xls (xlrd); .xlsb (pyxlsb)",
        "PowerPoint": ", ".join(sorted(_POWERPOINT)) + " (standard library)",
        "PDF": ".pdf (pypdf; existing text layer only)",
        "OpenDocument": ", ".join(sorted(_ODF)) + " (standard library)",
        "Other": ".ipynb, .html/.htm/.xhtml (standard library); .rtf (striprtf)",
        "Text": ", ".join(sorted(e for e in _TEXT if e)) + ", extensionless text",
        "Archives": ".zip, including supported documents and nested ZIPs",
        "Unsupported legacy/native": ".doc/.dot/.ppt/.pps/.pot, Apple .pages/.numbers/.key, .one/.mdb/.accdb, .7z/.rar/.tar/.gz; convert/export first",
    }


_HELP = {
    "quickstart": """
STRING FINDER -- notebook quick start

    import string_finder as sf
    overview = sf.precheck('~/Documents')
    overview.show()
    results = sf.search('~/Documents', 'family meeting')
    results.show()
    results.files                 # unique matching paths
    results.summary               # counts and elapsed seconds
    results.issues                # coverage gaps -- inspect these!
    results.to_dataframe()        # optional pandas table
    results.to_csv('~/Desktop/string_matches.csv')
    results.issues_to_csv('~/Desktop/string_issues.csv')

Defaults: recursive, case insensitive, ZIPs included, whitespace normalized,
hidden files omitted. Literal strings: no regex syntax to escape.

    sf.search('~/Documents', 'Family Meeting', case_sensitive=True)
    sf.search('~/Documents', 'couch', recursive=False, search_zips=False)
    sf.search('~/Documents', 'couch', include_extensions=['docx', 'pdf'])
    sf.search('~/Documents', 'couch', exclude_extensions=['csv', 'log'])
    sf.precheck('~/Documents', large_file_mb=10, search_zips=False).show()

Use sf.help('setup'/'formats'/'options'/'limits'/'cli') for more detail.
Built-in Python help(sf.search) shows the full signature and parameter docs.
""",
    "setup": """
SETUP -- same module from any notebook directory, macOS / Windows / Linux

1. Save string_finder.py in your existing helper scripts directory.
   If that directory is already on sys.path, just import string_finder as sf.

2. Install into the active notebook kernel (not a different terminal Python):

       %pip install tqdm pypdf openpyxl
       # Optional formats / table display:
       %pip install xlrd pyxlsb striprtf pandas

3. If your helper directory isn't registered, use its ABSOLUTE path:

       import sys
       from pathlib import Path
       helper_dir = str(Path('~/helper_scripts').expanduser().resolve())
       if helper_dir not in sys.path:
           sys.path.insert(0, helper_dir)
       import string_finder as sf

   Change '~/helper_scripts' to your real directory. Don't use os.chdir().
   Once this helper directory is on the import path, all searches can target
   absolute paths, regardless of where the notebook is stored.

Optional persistent registration (run once PER Python/kernel environment):

       import site
       from pathlib import Path
       helper_dir = Path('~/helper_scripts').expanduser().resolve()
       if not (helper_dir / 'string_finder.py').is_file():
           raise FileNotFoundError(helper_dir / 'string_finder.py')
       user_site = Path(site.getusersitepackages())
       user_site.mkdir(parents=True, exist_ok=True)
       (user_site / 'personal_helpers.pth').write_text(
           str(helper_dir) + '\\n', encoding='utf-8')

   Restart the kernel. Some virtual environments disable the user site; use
   the explicit sys.path block there. Existing registration needs no changes.
   To undo the optional registration, remove personal_helpers.pth.

If tqdm uses a plain text bar, it still works; widget support is optional.
Python >= 3.10. No Office installation required. No network used by searches.
""",
    "formats": """
FORMATS AND WHAT IS SEARCHED

Word OOXML: current text in body, tables/text boxes, headers/footers, footnotes,
endnotes, comments. Run fragments are joined within paragraphs. Deleted and
moved-from tracked revisions are excluded and flagged. Macro files are read
as text packages; macros are never executed.

Excel OOXML: all worksheets, including hidden ones; sheet names, individual
cell values, formula source and saved cached formula values, plus standard
comments. Formulas are not recalculated; missing caches are flagged. Dates
and numbers use Python's text representation, not Excel's display formatting.
.xls / .xlsb read saved cell values and sheet names via optional dependencies;
their formula source, comments, and formatting are outside this version.

PowerPoint OOXML: all slides, including hidden ones; paragraphs in shapes and
tables, speaker notes, standard comments, cached chart strings. Slide numbers
follow presentation order. Text inherited only from masters/layouts, images,
SmartArt, and embedded workbooks/objects are outside this version.

PDF: page-by-page existing text layer. PDFs that are image-only need OCR first.
Pages with no extracted text are flagged (they may also just be blank).
Mixed pages with images AND text can still have unsearched image text.

Other: plain text/CSV/JSON/XML/code, notebook source and text/plain outputs,
HTML visible text (scripts/styles omitted), RTF with striprtf. OpenDocument
content.xml text and selected spreadsheet attributes; layout not reconstructed.
Text formats are extension-based, with binary detection, BOM/UTF-8 detection,
and reported Windows-1252 fallback. Set encoding= for other encodings.

ZIP: members use their real format extractor; no raw-byte searching of Office
containers. Results use outer.zip!/folder/member.ext; nested paths are retained.
RAR, 7z, tar/gzip, encrypted documents, legacy .doc/.ppt, OneNote/Access, and
Apple iWork native formats require export/conversion first and are reported.

No OCR, embedded attachments, deleted revisions, filesystem metadata, file
names, PDF annotations, or universal Office object coverage is promised.
""",
    "options": """
OPTIONS -- search(directory, query, **options)

recursive=True                  filesystem subfolders (ZIP traversal independent)
case_sensitive=False            Unicode-aware case-insensitive literal matching
include_extensions=None         leaf extension allowlist; ['docx', '.pdf']
exclude_extensions=None         denylist; wins over include; '.zip' blocks traversal
search_zips=True                ZIP contents, including nested archives
progress=True                   tqdm.auto discovery counter, no fixed total/ETA
normalize_whitespace=True       collapse whitespace within query/text units
max_matches_per_file=None       matching-unit cap; reached caps are flagged
context_chars=90                snippet context on each side of first match
include_hidden=False            dot paths omitted; symlinks never followed
max_file_mb=250                 max file size (MiB), None removes this limit
max_zip_depth=3                 layers; 1=direct members only, 0=no ZIPs
max_archive_member_mb=250       uncompressed member/component size
max_archive_total_mb=2048       global ZIP expansion budget / per-Office package size
max_archive_ratio=1000          uncompressed/compressed size ratio limit
max_archive_members=100000      global ZIP inspection cap / per-Office entry cap
encoding=None                  auto BOM/UTF-8, then reported cp1252 fallback

Precheck accepts traversal/size/filter/encoding options plus large_file_mb=10
and progress=True. It does not accept the matching-specific options above.

Extension filters are case insensitive; dots are optional. 'docx,pdf' works.
None includes known types; [] includes no leaf documents. Include '.zip' only
does not select its contents; include the document extensions you want instead.
Units: plain-text line, Word paragraph, Excel cell, slide paragraph, PDF page,
notebook cell. A phrase doesn't span units. Snippets show first occurrence;
occurrences counts non-overlapping matches in that unit. Results retain all
matches by default. A file with matches may still have a coverage issue.
""",
    "limits": """
LIMITS, COVERAGE, AND PERFORMANCE

Inspect results.issues and results.summary. No hits means 'no hits in the text
successfully extracted', not proof of absence in unsearched images or objects.
Corrupt/encrypted/unsupported files, permission errors, decoding errors, size
limits, match caps, missing dependencies, and empty-text PDF pages are reported.
One bad file doesn't normally stop other files. Ctrl-C/Interrupt returns a
report marked interrupted with collected results. Memory exhaustion or a hard
kernel/process stop cannot be recovered by this mechanism.

Hidden paths and Office lock/Apple metadata files are excluded by default.
Symlinks are not followed. recursive=False restricts filesystem traversal;
search_zips=False is the separate switch for ZIP traversal.

Plain text is streamed by line; most Office XML, notebooks, RTF, and HTML may
be materialized in memory. PDFs/Excel can also use substantial memory. File
size and ZIP limits are practical bounds, not a sandbox or hard RAM/time cap.
No per-file timeout or worker-process isolation is included in this version.
ZIPs are never extracted into the searched tree. Large selected ZIP members
can spool to automatically removed system temporary files.

Default size limits can skip legitimate large files. Check the precheck and
issues, then raise only the needed limit (sizes are MiB, 1 MiB=1,048,576 bytes):

    sf.search(folder, phrase, max_file_mb=500, max_archive_member_mb=500)

Precheck includes ordinary ZIP directory inspection; nested ZIPs require
decompression. ZIP uncompressed size sums overlap for nested layers and are
not a disk-use estimate. Precheck is not a promise that every candidate can be
decoded or that a directory will remain unchanged before the next search.
Progress is a growing discovery counter rather than a misleading fixed ETA.
""",
    "cli": """
TERMINAL (optional; notebook import is the primary interface)

    python /absolute/path/string_finder.py --help
    python /absolute/path/string_finder.py ~/Documents 'family meeting'
    python /absolute/path/string_finder.py ~/Documents --precheck
    python /absolute/path/string_finder.py ~/Documents couch --include docx pdf
    python /absolute/path/string_finder.py ~/Documents couch --no-recursive --no-zips
    python /absolute/path/string_finder.py ~/Documents couch --csv ~/Desktop/hits.csv

Exit codes: 0 completed with no coverage issues; 1 completed with coverage issues;
2 invalid arguments/setup; 130 interrupted. A match itself doesn't change code.
""",
}


def help(topic: str | None = None) -> None:
    """Print built-in guide. Topics: setup, formats, options, limits, cli, all.

    sf.help() is the quick start; Python's help(sf.search) also works.
    """
    topic = "quickstart" if topic is None else str(topic).lower().strip()
    if topic == "all":
        for value in _HELP.values():
            print(value.strip() + "\n")
    elif topic in _HELP:
        print(_HELP[topic].strip())
        if topic == "formats":
            for name, description in supported_formats().items():
                print(f"\n{name}: {description}")
    else:
        raise ValueError(f"Unknown help topic {topic!r}. Choose: {', '.join(_HELP)}, all")


def _main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                     epilog="Notebook guide: import string_finder as sf; sf.help('all')")
    parser.add_argument("directory", nargs="?", help="Directory or file (default: current directory)")
    parser.add_argument("query", nargs="?", help="Literal string/phrase; quote phrases in your shell")
    parser.add_argument("--precheck", action="store_true", help="Inventory instead of searching")
    parser.add_argument("--guide", choices=[*list(_HELP), "all"], help="Print built-in guide and exit")
    parser.add_argument("--no-recursive", action="store_true", help="Don't enter filesystem subfolders")
    parser.add_argument("--case-sensitive", action="store_true")
    parser.add_argument("--include", nargs="+", help="Leaf extensions to include, e.g. docx pdf")
    parser.add_argument("--exclude", nargs="+", help="Extensions to exclude")
    parser.add_argument("--no-zips", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--include-hidden", action="store_true")
    parser.add_argument("--exact-whitespace", action="store_true")
    parser.add_argument("--max-file-mb", type=float, default=250)
    parser.add_argument("--max-zip-depth", type=int, default=3)
    parser.add_argument("--encoding")
    parser.add_argument("--large-file-mb", type=float, default=10, help="Precheck threshold (MiB)")
    parser.add_argument("--limit", type=int, default=20, help="Display limit only; all hits remain stored/exported")
    parser.add_argument("--csv", help="Export matching units to CSV")
    parser.add_argument("--issues-csv", help="Export coverage issues to CSV")
    args = parser.parse_args(argv)
    if args.guide:
        help(args.guide)
        return 0
    if not args.precheck and args.query is None:
        parser.error("query is required unless --precheck or --guide is used")
    options = dict(recursive=not args.no_recursive, include_extensions=args.include,
                   exclude_extensions=args.exclude, search_zips=not args.no_zips,
                   progress=not args.no_progress, include_hidden=args.include_hidden,
                   max_file_mb=args.max_file_mb, max_zip_depth=args.max_zip_depth,
                   encoding=args.encoding)
    try:
        if args.precheck:
            if args.csv or args.issues_csv:
                parser.error("--csv and --issues-csv are search options; precheck returns an inventory")
            report = precheck(args.directory or ".", large_file_mb=args.large_file_mb, **options)
            report.show(limit=args.limit)
        else:
            report = search(args.directory or ".", args.query, case_sensitive=args.case_sensitive,
                            normalize_whitespace=not args.exact_whitespace, **options)
            report.show(limit=args.limit)
            if args.csv:
                report.to_csv(args.csv)
            if args.issues_csv:
                report.issues_to_csv(args.issues_csv)
    except (ValueError, OSError, TypeError) as exc:
        parser.error(str(exc))
    return 130 if report.interrupted else (1 if report.issues else 0)


if __name__ == "__main__":
    raise SystemExit(_main())
