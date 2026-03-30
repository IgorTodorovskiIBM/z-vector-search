#!/usr/bin/env python3
"""
Convert IBM z/OS MVS System Codes PDF text extract into clean per-code text files
suitable for z-index --ibm-messages ingestion.

Each system completion code (abend code) becomes one chunk with a synthetic
message ID of the form "ABEND_xxx" (e.g. ABEND_0C4, ABEND_001).
Wait state codes become "WAIT_xxx" and IPCS codes become "IPCS_xxxx".

The 0Cx family (program interrupts S0C1-S0CF) gets special handling:
each sub-code is emitted as its own entry for better search results.

Usage:
    # Step 1: Extract text from PDF
    pdftotext ibm-docs/mvs-system-codes.pdf ibm-docs/txt/mvs-system-codes.txt
    # Step 2: Run this script
    python3 tools/pdf-to-abendcodes.py ibm-docs/txt/mvs-system-codes.txt ibm-docs/clean/
"""

import sys
import os
import re

# A top-level system code: 3 hex digits alone on a line
SYSTEM_CODE_RE = re.compile(r'^([0-9A-F]{3})$')

# Wildcard codes like 0Cx, 09x, FFx
WILDCARD_CODE_RE = re.compile(r'^([0-9A-F]{2}[xX])$')

# Sub-codes within 0Cx: "0C4" at start of line followed by text
SUBCODE_0CX_RE = re.compile(r'^(0C[0-9A-F])\s*$')

# IPCS user completion codes are 4 decimal digits
IPCS_CODE_RE = re.compile(r'^(\d{4})$')

# Chapter boundaries
CHAPTER_RE = re.compile(r'^Chapter (\d+)\.\s+(.*?)(?:\s+\d+)?$')

# Lines to skip (page headers, footers, etc.)
SKIP_PATTERNS = [
    re.compile(r'^Chapter \d+\.\s+.*\d+$'),
    re.compile(r'^\d+\s+z/OS:'),
    re.compile(r'^z/OS:\s'),
    re.compile(r'^©\s*Copyright'),
    re.compile(r'^IBM\s*$'),
    re.compile(r'^SA\d{2}-\d{4}'),
    re.compile(r'^\x0c'),
    re.compile(r'^z/OS$'),
    re.compile(r'^Contents$'),
    re.compile(r'^Index$'),
    re.compile(r'^Notices\.*\s*\d*$'),
    re.compile(r'^Appendix [A-Z]\.'),
    re.compile(r'^About this document'),
    re.compile(r'^How to provide feedback'),
    re.compile(r'^Summary of changes'),
    re.compile(r'^Tables\.*\s*\d*$'),
]

def should_skip(line):
    for pat in SKIP_PATTERNS:
        if pat.search(line):
            return True
    return False

def is_toc_line(line):
    return '...' in line


def split_0cx_entry(code_id, text):
    """Split the 0Cx entry into individual sub-code entries (0C1, 0C2, ..., 0CF).

    The 0Cx entry has a shared explanation header, then sub-codes like:
        0C1
        Operation exception. The reason code is 1.
        0C4
        One of the following exceptions or errors occurred:
        4
        Protection exception...
    And a shared System action/Source footer.

    We emit:
    1. The parent entry (ABEND_0Cx) with the full text
    2. Individual entries (ABEND_0C1, ABEND_0C4, etc.) with their specific text
       plus the shared header/footer
    """
    lines = text.split('\n')
    entries = []

    # Find the shared header (everything before first sub-code "0Cx" line)
    # and shared footer (System action, Source, etc. after last sub-code)
    header_lines = []
    footer_lines = []
    subcodes = []  # list of (code, [lines])

    current_subcode = None
    current_sub_lines = []
    in_footer = False

    # Sub-code pattern: "0C1", "0C2", ..., "0CF" alone on a line or followed by short text
    subcode_re = re.compile(r'^(0C[0-9A-F])$')

    # Footer starts at "System action" after all subcodes
    footer_keywords = ['System action', 'System programmer response',
                       'Programmer response', 'Operator response', 'Source']

    def flush_sub():
        nonlocal current_subcode, current_sub_lines
        if current_subcode and current_sub_lines:
            while current_sub_lines and not current_sub_lines[-1].strip():
                current_sub_lines.pop()
            subcodes.append((current_subcode, list(current_sub_lines)))
        current_subcode = None
        current_sub_lines = []

    for line in lines:
        stripped = line.strip()

        if in_footer:
            footer_lines.append(line)
            continue

        m = subcode_re.match(stripped)
        if m and not in_footer:
            flush_sub()
            current_subcode = m.group(1)
            current_sub_lines = []
            continue

        if current_subcode is None:
            # Still in header
            header_lines.append(line)
        else:
            # Check if we hit the shared footer
            if stripped in footer_keywords or (stripped.startswith('System action') and not current_sub_lines):
                flush_sub()
                in_footer = True
                footer_lines.append(line)
            else:
                current_sub_lines.append(line)

    flush_sub()

    # Emit the parent entry with full text
    entries.append((code_id, text))

    # Emit individual sub-code entries
    shared_header = '\n'.join(header_lines).strip()
    shared_footer = '\n'.join(footer_lines).strip()

    for subcode, sub_lines in subcodes:
        sub_text_body = '\n'.join(sub_lines).strip()
        if not sub_text_body:
            continue

        parts = [f"System abend code {subcode} (S{subcode})"]
        if sub_text_body:
            parts.append(sub_text_body)
        if shared_footer:
            parts.append(shared_footer)
        sub_text = '\n\n'.join(parts)
        entries.append((f"ABEND_{subcode}", sub_text))

    return entries


def parse_system_codes(lines):
    """Parse the text into (code_id, text) tuples."""
    codes = []
    current_code = None
    current_prefix = None
    current_lines = []
    in_chapter = 0
    in_preamble = True

    def flush():
        nonlocal current_code, current_lines
        if current_code and current_lines:
            while current_lines and not current_lines[-1].strip():
                current_lines.pop()
            if current_lines:
                text = '\n'.join(current_lines)
                # Special handling for 0Cx — split into sub-entries
                if current_code == 'ABEND_0Cx':
                    codes.extend(split_0cx_entry(current_code, text))
                else:
                    codes.append((current_code, text))
        current_code = None
        current_lines = []

    def is_new_code(line, i_pos):
        """Check if this line starts a new top-level code entry."""
        # Must be followed by "Explanation" (after skipping blanks and page headers)
        j = i_pos + 1
        while j < len(lines):
            stripped = lines[j].strip()
            if not stripped or should_skip(stripped):
                j += 1
                continue
            return stripped.startswith('Explanation')
        return False

    i = 0
    while i < len(lines):
        line = lines[i].rstrip().replace('\x0c', '')

        # Detect chapter boundaries
        m_ch = CHAPTER_RE.match(line)
        if m_ch:
            ch_num = int(m_ch.group(1))
            if ch_num == 2:
                in_chapter = 2
                current_prefix = "ABEND_"
                in_preamble = False
            elif ch_num == 3:
                flush()
                in_chapter = 3
                current_prefix = "WAIT_"
            elif ch_num == 6:
                flush()
                in_chapter = 6
                current_prefix = "SADUMP_WAIT_"
            elif ch_num == 7:
                flush()
                in_chapter = 7
                current_prefix = "IPCS_"
            elif ch_num in (4, 5):
                flush()
                in_chapter = ch_num
            else:
                flush()
                in_chapter = ch_num
            i += 1
            continue

        if in_preamble:
            i += 1
            continue

        if in_chapter in (4, 5) or in_chapter > 7:
            i += 1
            continue

        if should_skip(line) or is_toc_line(line):
            i += 1
            continue

        # Check for new system/wait code (exact 3 hex digits)
        if in_chapter in (2, 3, 6) and current_prefix:
            m = SYSTEM_CODE_RE.match(line)
            if m and is_new_code(line, i):
                flush()
                current_code = current_prefix + m.group(1)
                current_lines = [f"System code {m.group(1)} (S{m.group(1)})"]
                i += 1
                continue

            # Wildcard codes like 0Cx, 09x, FFx
            m_w = WILDCARD_CODE_RE.match(line)
            if m_w and is_new_code(line, i):
                flush()
                current_code = current_prefix + m_w.group(1)
                current_lines = [f"System code {m_w.group(1)} (S{m_w.group(1)})"]
                i += 1
                continue

            # Not a top-level code — append as text
            if current_code is not None:
                current_lines.append(line)
                i += 1
                continue

        # Check for IPCS code
        if in_chapter == 7 and current_prefix:
            m = IPCS_CODE_RE.match(line)
            if m and is_new_code(line, i):
                flush()
                current_code = current_prefix + m.group(1)
                current_lines = [f"IPCS user completion code {m.group(1)}"]
                i += 1
                continue

        # Regular text line
        if current_code is not None:
            current_lines.append(line)

        i += 1

    flush()
    return codes


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <input_file> <output_dir>")
        print(f"  input_file: text file from pdftotext of MVS System Codes")
        print(f"  output_dir: directory for cleaned output files")
        sys.exit(1)

    input_file = sys.argv[1]
    output_dir = sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)

    with open(input_file, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    codes = parse_system_codes(lines)

    if not codes:
        print("WARNING: No codes found!")
        return

    # Count by type
    abend_count = sum(1 for c, _ in codes if c.startswith('ABEND_'))
    wait_count = sum(1 for c, _ in codes if c.startswith('WAIT_') or c.startswith('SADUMP_'))
    ipcs_count = sum(1 for c, _ in codes if c.startswith('IPCS_'))

    # Write output file in ibm-messages format
    output_path = os.path.join(output_dir, "mvs-system-codes.txt")
    with open(output_path, 'w', encoding='utf-8') as f:
        for code_id, text in codes:
            f.write(f"{code_id}\n")
            f.write(text)
            f.write('\n\n')

    print(f"Extracted {len(codes)} codes:")
    print(f"  System completion (abend) codes: {abend_count}")
    print(f"  Wait state codes: {wait_count}")
    print(f"  IPCS user completion codes: {ipcs_count}")
    print(f"Output: {output_path}")


if __name__ == '__main__':
    main()
