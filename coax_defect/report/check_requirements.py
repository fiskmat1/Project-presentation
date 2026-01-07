import os
import re
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
TEX_PATH = os.path.join(HERE, 'main.tex')
BIB_PATH = os.path.join(HERE, 'biblio.bib')
PDF_PATH = os.path.join(HERE, 'main.pdf')


def read_text(path):
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        return f.read()


def unique_cite_keys(tex):
    keys = set()
    for m in re.finditer(r'\\cite\{([^}]+)\}', tex):
        for part in m.group(1).split(','):
            key = part.strip()
            if key:
                keys.add(key)
    return keys


def count_env(tex, env_name):
    pat = r'\\begin\{' + re.escape(env_name) + r'\*?\}'
    return len(re.findall(pat, tex))


def labels_in_env(tex, env_name):
    begin = r'\\begin\{' + re.escape(env_name) + r'\*?\}'
    end = r'\\end\{' + re.escape(env_name) + r'\*?\}'
    labels = set()
    for m in re.finditer(begin + r'(.*?)' + end, tex, flags=re.S):
        block = m.group(1)
        for lab in re.findall(r'\\label\{([^}]+)\}', block):
            labels.add(lab)
    return labels


def labels_referenced(tex, labels):
    used = set()
    for lab in labels:
        if re.search(r'\\ref\{' + re.escape(lab) + r'\}', tex):
            used.add(lab)
    return used


def parse_bib_entries(bib_text):
    entries = {}
    pat = r'@(\w+)\s*\{\s*([^,]+)\s*,(.*?)\n\}'
    for m in re.finditer(pat, bib_text, flags=re.S):
        entry_type = m.group(1).strip().lower()
        key = m.group(2).strip()
        body = m.group(3)
        entries[key] = {'type': entry_type, 'body': body}
    return entries


def bib_has_field(body, field):
    return re.search(r'(?im)^\s*' + re.escape(field) + r'\s*=\s*\{', body) is not None


def pdf_page_count(pdf_path):
    if not os.path.exists(pdf_path):
        return None, 'No PDF found (main.pdf).'
    try:
        out = subprocess.check_output(['pdfinfo', pdf_path], text=True, stderr=subprocess.DEVNULL)
        m = re.search(r'(?im)^Pages:\s*(\d+)\s*$', out)
        if m:
            return int(m.group(1)), 'Counted pages using pdfinfo.'
    except Exception:
        pass
    for mod in ('pypdf', 'PyPDF2'):
        try:
            lib = __import__(mod)
            reader = lib.PdfReader(pdf_path)
            return len(reader.pages), 'Counted pages using ' + mod + '.'
        except Exception:
            continue
    return None, "PDF exists, but couldn't count pages (no pdfinfo / pypdf / PyPDF2)."


def main():
    problems = []

    tex = read_text(TEX_PATH)
    bib = read_text(BIB_PATH)
    bib_entries = parse_bib_entries(bib)

    n_fig = count_env(tex, 'figure')
    n_tab = count_env(tex, 'table')
    cite_keys = unique_cite_keys(tex)

    fig_labels = labels_in_env(tex, 'figure')
    tab_labels = labels_in_env(tex, 'table')
    all_labels = fig_labels | tab_labels
    referenced = labels_referenced(tex, all_labels)

    missing_refs = sorted(all_labels - referenced)
    missing_bib = sorted([k for k in cite_keys if k not in bib_entries])
    bib_missing_note = sorted([k for k, e in bib_entries.items() if not bib_has_field(e['body'], 'note')])

    bib_articles_missing_doi_url = []
    for k, e in bib_entries.items():
        if e['type'] == 'article':
            if not (bib_has_field(e['body'], 'doi') and bib_has_field(e['body'], 'url')):
                bib_articles_missing_doi_url.append(k)
    bib_articles_missing_doi_url.sort()

    print('=== Report requirement checks ===')
    print('Figures:', n_fig)
    print('Tables: ', n_tab)
    print('Unique citation keys used (\\cite{...}):', len(cite_keys))
    print()

    if missing_refs:
        problems.append('Unreferenced labels (figure/table): ' + ', '.join(missing_refs))
    if missing_bib:
        problems.append('Cited keys missing from biblio.bib: ' + ', '.join(missing_bib))
    if bib_missing_note:
        problems.append("Bib entries missing mandatory 'note' field: " + ', '.join(bib_missing_note))
    if bib_articles_missing_doi_url:
        problems.append('@article entries missing doi/url: ' + ', '.join(bib_articles_missing_doi_url))

    pages, msg = pdf_page_count(PDF_PATH)
    if pages is None:
        print('PDF pages: (unknown) —', msg)
    else:
        print('PDF pages:', pages, '—', msg)

    if problems:
        print('\nProblems found:')
        for p in problems:
            print('-', p)
        return 2

    print('\nNo problems found by this script.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
