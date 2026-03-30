#!/usr/bin/env python3
"""
Convert IBM z/OS Messages PDF text extracts into clean per-message text files
suitable for z-index --ibm-messages ingestion.

Usage:
    # Step 1: Extract text from PDFs (done separately with pdftotext)
    # Step 2: Run this script to clean and split
    python3 tools/pdf-to-messages.py ibm-docs/txt/ ibm-docs/clean/
"""

import sys
import os
import re

# Pattern for a message ID at the start of a line (standalone or followed by text)
# 2-8 uppercase alpha (including $#@) + 1-5 digits + optional severity letter
MSGID_STANDALONE_RE = re.compile(r'^([A-Z$#@]{2,8}\d{1,5}[IEWADSX]?)\s*$')
# Message ID followed by message text on same line (e.g. RACF 2-column PDFs)
MSGID_INLINE_RE = re.compile(r'^([A-Z$#@]{2,8}\d{1,5}[IEWADSX]?)\s+(\S.*)$')

# Lines to skip (page headers, footers, copyright, etc.)
SKIP_PATTERNS = [
    re.compile(r'^Chapter \d+\.\s+\w+ messages\s+\d+'),        # "Chapter 2. IEF messages 31"
    re.compile(r'^Chapter \d+\.\s'),                             # "Chapter N. ..."
    re.compile(r'^\d+\s+z/OS:'),                                # "30  z/OS: z/OS MVS System Messages..."
    re.compile(r'^z/OS:\s'),                                     # "z/OS: z/OS MVS System Messages..."
    re.compile(r'^©\s*Copyright'),                               # Copyright lines
    re.compile(r'^IBM\s*$'),                                     # Standalone "IBM"
    re.compile(r'^SA\d{2}-\d{4}'),                               # Publication numbers
    re.compile(r'^\x0c'),                                        # Form feed characters
    re.compile(r'^z/OS$'),                                       # Standalone "z/OS"
    re.compile(r'^\d+\s*$'),                                     # Standalone page numbers
    re.compile(r'^Contents$'),
    re.compile(r'^Index$'),
    re.compile(r'^Notices\.*\s*\d*$'),
    re.compile(r'^Appendix [A-Z]\.'),
    re.compile(r'^About this document'),
    re.compile(r'^How to provide feedback'),
    re.compile(r'^Summary of changes'),
    re.compile(r'^Summary of message changes'),
    re.compile(r'^Message changes for'),
    re.compile(r'^\s*New$'),
    re.compile(r'^\s*Changed$'),
    re.compile(r'^\s*Deleted$'),
    re.compile(r'^The following messages are'),
    re.compile(r'^Who uses MVS'),
]

def should_skip(line):
    """Check if a line is PDF noise that should be stripped."""
    for pat in SKIP_PATTERNS:
        if pat.search(line):
            return True
    return False

def is_toc_line(line):
    """Detect table-of-contents lines like 'Chapter 2. IEF messages...... 29'"""
    return '...' in line

def clean_text(lines):
    """Remove PDF artifacts from extracted text."""
    cleaned = []
    in_toc = False
    in_preamble = True

    for line in lines:
        stripped = line.rstrip()

        # Skip form feeds
        if '\x0c' in stripped:
            stripped = stripped.replace('\x0c', '')

        # Detect end of preamble (first real message ID)
        if in_preamble and (MSGID_STANDALONE_RE.match(stripped) or MSGID_INLINE_RE.match(stripped)):
            # Check if this could be a real message (not in TOC/summary)
            # Real messages are followed by their text on subsequent lines
            in_preamble = False

        if in_preamble:
            # Still in front matter - look for chapter start markers
            if stripped.startswith('Chapter') and 'messages' in stripped and '...' not in stripped:
                in_preamble = False
            continue

        # Skip TOC lines
        if is_toc_line(stripped):
            continue

        # Skip noise patterns
        if should_skip(stripped):
            continue

        cleaned.append(stripped)

    return cleaned

def split_messages(lines, source_filename):
    """Split cleaned text into individual message entries."""
    messages = []
    current_msgid = None
    current_lines = []

    def flush():
        if current_msgid and current_lines:
            # Trim trailing empty lines
            while current_lines and not current_lines[-1].strip():
                current_lines.pop()
            if current_lines:
                text = '\n'.join(current_lines)
                messages.append((current_msgid, text))

    for line in lines:
        m = MSGID_STANDALONE_RE.match(line)
        if m:
            flush()
            current_msgid = m.group(1)
            current_lines = []
        else:
            m2 = MSGID_INLINE_RE.match(line)
            if m2 and current_msgid != m2.group(1):
                # New message with text on same line (common in 2-column PDF extracts)
                # Only treat as new if the remaining text looks like a message
                # (multiple words, not just a module name)
                rest = m2.group(2)
                if len(rest.split()) >= 3:
                    flush()
                    current_msgid = m2.group(1)
                    current_lines = [rest]
                elif current_msgid is not None:
                    current_lines.append(line)
            elif current_msgid is not None:
                current_lines.append(line)

    flush()
    return messages

def process_file(input_path, output_dir):
    """Process a single text file extracted from an IBM messages PDF."""
    basename = os.path.splitext(os.path.basename(input_path))[0]

    with open(input_path, 'r', encoding='utf-8', errors='replace') as f:
        raw_lines = f.readlines()

    # Clean the text
    cleaned = clean_text(raw_lines)

    # Split into messages
    messages = split_messages(cleaned, basename)

    if not messages:
        print(f"  WARNING: No messages found in {input_path}")
        return 0

    # Write output file: one message per entry, separated by msgid boundaries
    # Format compatible with z-index --ibm-messages
    output_path = os.path.join(output_dir, f"{basename}.txt")
    with open(output_path, 'w', encoding='utf-8') as f:
        for msgid, text in messages:
            f.write(f"{msgid}\n")
            f.write(text)
            f.write('\n\n')

    return len(messages)

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <input_dir> <output_dir>")
        print(f"  input_dir:  directory with .txt files from pdftotext")
        print(f"  output_dir: directory for cleaned output files")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)

    total = 0
    for fname in sorted(os.listdir(input_dir)):
        if not fname.endswith('.txt'):
            continue
        input_path = os.path.join(input_dir, fname)
        count = process_file(input_path, output_dir)
        total += count
        print(f"  {fname}: {count} messages")

    print(f"\nTotal: {total} messages extracted")

if __name__ == '__main__':
    main()
