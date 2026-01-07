#!/usr/bin/env python3
"""
Lightweight compliance checks for the course report requirements.

This script is intentionally dependency-light (pure stdlib) and checks:
- number of figures/tables in `main.tex`
- that each figure/table label is referenced in the text
- number of unique citation keys used via \\cite{...}
- that cited keys exist in `biblio.bib`
- that each bib entry has a mandatory `note` field
- that `@article` entries have `doi` and `url` fields

It optionally reports PDF page count if `main.pdf` exists and either `pdfinfo`
or a PDF python library is available.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TEX = ROOT / "main.tex"
BIB = ROOT / "biblio.bib"
PDF = ROOT / "main.pdf"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _unique_cite_keys(tex: str) -> set[str]:
    keys: set[str] = set()
    for m in re.finditer(r"\\cite\{([^}]+)\}", tex):
        for k in m.group(1).split(","):
            k = k.strip()
            if k:
                keys.add(k)
    return keys


def _count_env(tex: str, env: str) -> int:
    # counts \begin{figure} and \begin{figure*} etc
    return len(re.findall(rf"\\begin\{{{re.escape(env)}\*?\}}", tex))


def _labels_in_env(tex: str, env: str) -> set[str]:
    labels: set[str] = set()
    # very simple: find blocks between begin/end, extract \label{...}
    for m in re.finditer(rf"\\begin\{{{re.escape(env)}\*?\}}(.*?)\\end\{{{re.escape(env)}\*?\}}", tex, flags=re.S):
        block = m.group(1)
        labels.update(re.findall(r"\\label\{([^}]+)\}", block))
    return labels


def _labels_referenced(tex: str, labels: set[str]) -> set[str]:
    referenced: set[str] = set()
    for lab in labels:
        if re.search(rf"\\ref\{{{re.escape(lab)}\}}", tex):
            referenced.add(lab)
    return referenced


@dataclass(frozen=True)
class BibEntry:
    entry_type: str
    key: str
    body: str

    def has_field(self, field: str) -> bool:
        # matches "field = {..}" or "field={..}" etc (case-insensitive)
        return re.search(rf"(?im)^\s*{re.escape(field)}\s*=\s*\{{", self.body) is not None


def _parse_bib(bib: str) -> dict[str, BibEntry]:
    entries: dict[str, BibEntry] = {}
    # naive parser: @type{key, ...}\n
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,]+)\s*,(.*?)\n\}", bib, flags=re.S):
        entry_type = m.group(1).strip().lower()
        key = m.group(2).strip()
        body = m.group(3)
        entries[key] = BibEntry(entry_type=entry_type, key=key, body=body)
    return entries


def _pdf_page_count(pdf: Path) -> tuple[int | None, str]:
    if not pdf.exists():
        return None, "No PDF found (main.pdf)."

    # 1) try pdfinfo
    import subprocess

    try:
        out = subprocess.check_output(["pdfinfo", str(pdf)], text=True, stderr=subprocess.DEVNULL)
        m = re.search(r"(?im)^Pages:\s*(\d+)\s*$", out)
        if m:
            return int(m.group(1)), "Counted pages using pdfinfo."
    except Exception:
        pass

    # 2) try python libs
    for mod in ("pypdf", "PyPDF2"):
        try:
            lib = __import__(mod)
            reader = lib.PdfReader(str(pdf))
            return len(reader.pages), f"Counted pages using {mod}."
        except Exception:
            continue

    return None, "PDF exists, but couldn't count pages (no pdfinfo / pypdf / PyPDF2)."


def main() -> int:
    problems: list[str] = []

    tex = _read(TEX)
    bib = _read(BIB)
    bib_entries = _parse_bib(bib)

    n_fig = _count_env(tex, "figure")
    n_tab = _count_env(tex, "table")
    cite_keys = _unique_cite_keys(tex)

    fig_labels = _labels_in_env(tex, "figure")
    tab_labels = _labels_in_env(tex, "table")
    all_labels = fig_labels | tab_labels
    referenced = _labels_referenced(tex, all_labels)
    missing_refs = sorted(all_labels - referenced)

    missing_bib = sorted(k for k in cite_keys if k not in bib_entries)

    # Bib checks
    bib_missing_note = sorted(k for k, e in bib_entries.items() if not e.has_field("note"))
    bib_articles_missing_doi_url: list[str] = []
    for k, e in bib_entries.items():
        if e.entry_type == "article":
            if not (e.has_field("doi") and e.has_field("url")):
                bib_articles_missing_doi_url.append(k)
    bib_articles_missing_doi_url.sort()

    # Report
    print("=== Report requirement checks ===")
    print(f"Figures: {n_fig}")
    print(f"Tables:  {n_tab}")
    print(f"Unique citation keys used (\\cite{{...}}): {len(cite_keys)}")
    print()

    if missing_refs:
        problems.append(f"Unreferenced labels (figure/table): {', '.join(missing_refs)}")
    if missing_bib:
        problems.append(f"Cited keys missing from biblio.bib: {', '.join(missing_bib)}")
    if bib_missing_note:
        problems.append(f"Bib entries missing mandatory 'note' field: {', '.join(bib_missing_note)}")
    if bib_articles_missing_doi_url:
        problems.append(f"@article entries missing doi/url: {', '.join(bib_articles_missing_doi_url)}")

    pages, msg = _pdf_page_count(PDF)
    if pages is None:
        print(f"PDF pages: (unknown) — {msg}")
    else:
        print(f"PDF pages: {pages} — {msg}")

    if problems:
        print("\nProblems found:")
        for p in problems:
            print(f"- {p}")
        return 2

    print("\nNo problems found by this script.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



