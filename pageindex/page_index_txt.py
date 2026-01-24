"""PageIndex support for plain text files with robust structure detection.

Detects headings via multiple heuristics to support various historical source formats:
- ALL CAPS lines (chapter/section titles)
- Roman numeral sections (I. Section, II. Section)
- Chapter markers (CHAPTER I, CHAPITRE II)
- Book/Part markers (FIRST BOOK, LIVRE PREMIER)
- Ordinal headings (PREMIER MIRACLE, DEUXIÈME CHAPITRE)
- Markdown headers (#, ##, ###)
- Separator-based structure (---, ===)
- Numbered paragraphs (§1, §2) for continuous texts

Handles OCR artifacts like multi-line headings, page markers, and Google Books metadata.
"""

import asyncio
import json
import os
import re

try:
    from .page_index_md import generate_summaries_for_structure_md
    from .utils import (
        count_tokens,
        create_clean_structure_for_description,
        format_structure,
        generate_doc_description,
        write_node_id,
    )
except ImportError:
    from page_index_md import generate_summaries_for_structure_md
    from utils import (
        create_clean_structure_for_description,
        format_structure,
        generate_doc_description,
        write_node_id,
    )


# OCR artifacts to skip (lowercase for matching)
OCR_ARTIFACTS = {
    "digitized by google",
    "digitized by",
    "dbygoqgl",
    "google",
    "a propos de ce livre",
    "about google book search",
    "usage guidelines",
    "consignes d'utilisation",
    "this is a digital copy",
    "ceci est une copie numerique",
    "copyright",
    "tous droits réservés",
    "all rights reserved",
}

# Patterns for OCR noise
OCR_NOISE_PATTERNS = [
    r'^\d+\s*$',  # Standalone page numbers
    r'^[\^\\\|/\*\-\.\,\;\:\'\"\s]+$',  # Punctuation-only lines
    r'^[a-zA-Z]\s*$',  # Single letters
    r'^\s*\d+\s*,\s*$',  # "123," patterns
    r'^http[s]?://',  # URLs
    r'^ark:/',  # Archive.org identifiers
]


def is_ocr_artifact(line: str) -> bool:
    """Check if line is an OCR artifact or noise."""
    stripped = line.strip().lower()
    if not stripped:
        return False

    # Direct matches
    if stripped in OCR_ARTIFACTS:
        return True

    # Pattern matches
    for pattern in OCR_NOISE_PATTERNS:
        if re.match(pattern, stripped):
            return True

    # Very short gibberish (OCR errors)
    if len(stripped) < 4 and not stripped.isalpha():
        return True

    return False


def is_all_caps_heading(line: str) -> bool:
    """Check if line is an all-caps heading.

    Supports French accented capitals and common punctuation.
    """
    stripped = line.strip()
    if len(stripped) < 5:
        return False

    # Allow uppercase letters (including French accented), spaces, hyphens, common punctuation
    # Pattern: starts with uppercase, contains mostly uppercase
    if re.match(r'^[A-ZÀÂÄÉÈÊËÏÎÔÙÛÜÇ][A-ZÀÂÄÉÈÊËÏÎÔÙÛÜÇ\s\-,\.\'\"()]+$', stripped):
        return True

    return False


def detect_markdown_header(line: str) -> tuple[bool, int, str]:
    """Detect markdown-style headers.

    Returns:
        Tuple of (is_header, level, title) where level is 1-6.
    """
    stripped = line.strip()

    match = re.match(r'^(#{1,6})\s+(.+)$', stripped)
    if match:
        level = len(match.group(1))
        title = match.group(2).strip()
        return True, level, title

    return False, 0, ""


def detect_separator(line: str) -> bool:
    """Detect markdown-style separators (---, ===, etc.)."""
    stripped = line.strip()

    if re.match(r'^[\-]{3,}$', stripped):
        return True
    if re.match(r'^[=]{3,}$', stripped):
        return True

    return False


def detect_roman_numeral_heading(line: str) -> tuple[bool, str]:
    """Detect Roman numeral section headings like 'I. Comment...' or 'XII. De la...'.

    Returns:
        Tuple of (is_heading, title)
    """
    stripped = line.strip()

    # Pattern: Roman numeral at start, followed by period/paren and text
    match = re.match(r'^([IVXLCDM]+)[\.\)]\s+(.+)$', stripped)
    if match:
        return True, stripped

    return False, ""


def detect_chapter_heading(line: str) -> tuple[bool, str, int]:
    """Detect chapter headings like 'CHAPTER I' or 'CHAPITRE II'.

    Returns:
        Tuple of (is_heading, title, level) where level indicates nesting.
    """
    stripped = line.strip()
    upper = stripped.upper()

    # English chapter markers
    if re.match(r'^CHAPTER\s+[IVXLCDM\d]+', upper):
        return True, stripped, 1

    # French chapter markers
    if re.match(r'^CHAPITRE\s+[IVXLCDM\d]+', upper):
        return True, stripped, 1

    return False, "", 0


def detect_book_heading(line: str) -> tuple[bool, str, int]:
    """Detect book/part headings.

    IMPORTANT: Requires the ENTIRE line to be a book marker to avoid
    false positives like "premier s'appelait" matching as PREMIER.

    Returns:
        Tuple of (is_heading, title, level)
    """
    stripped = line.strip()
    upper = stripped.upper()

    # Only match if the line is primarily a book/part marker (short and structural)
    if len(stripped) > 60:  # Too long to be a standalone marker
        return False, "", 0

    # English ordinals + BOOK/PART
    if re.match(r'^(FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH|NINTH|TENTH)\s+(BOOK|PART)\b', upper):
        return True, stripped, 1

    # French: LIVRE PREMIER, PARTIE PREMIÈRE, etc.
    if re.match(r'^LIVRE\s+(PREMIER|SECOND|DEUXIÈME|TROISIÈME|QUATRIÈME|CINQUIÈME)', upper):
        return True, stripped, 1
    if re.match(r'^(PREMIÈRE|DEUXIÈME|TROISIÈME|QUATRIÈME|CINQUIÈME)\s+PARTIE', upper):
        return True, stripped, 1

    # Direct BOOK/PART markers (e.g., "BOOK I", "PART 2")
    if re.match(r'^(BOOK|PART|LIVRE|PARTIE)\s+[IVXLCDM\d]+\b', upper):
        return True, stripped, 1

    return False, "", 0


def detect_ordinal_heading(line: str) -> tuple[bool, str, int]:
    """Detect ordinal headings like 'PREMIER MIRACLE' or 'DEUXIÈME CHAPITRE'.

    Returns:
        Tuple of (is_heading, title, level)
    """
    stripped = line.strip()
    upper = stripped.upper()

    # Only match short structural lines
    if len(stripped) > 60:
        return False, "", 0

    # French ordinals followed by a noun
    if re.match(r'^(PREMIER|PREMIÈRE|DEUXIÈME|TROISIÈME|QUATRIÈME|CINQUIÈME|SIXIÈME|SEPTIÈME|HUITIÈME|NEUVIÈME|DIXIÈME|ONZIÈME|DOUZIÈME)\s+(MIRACLE|CHAPITRE|LIVRE|PARTIE|SECTION|TABLEAU)', upper):
        return True, stripped, 2

    return False, "", 0


def detect_prologue_heading(line: str) -> tuple[bool, str, int]:
    """Detect prologue/preface/introduction headings.

    Returns:
        Tuple of (is_heading, title, level)
    """
    stripped = line.strip()
    upper = stripped.upper()

    if upper in ['PROLOGUE', 'PRÉFACE', 'PREFACE', 'AVERTISSEMENT', 'INTRODUCTION',
                 'AVANT-PROPOS', 'CONCLUSION', 'ÉPILOGUE', 'EPILOGUE']:
        return True, stripped, 2

    if re.match(r"^AVERTISSEMENT\s+DE\s+L['\u2019]", upper):
        return True, stripped, 2

    return False, "", 0


def detect_heading(line: str) -> dict:
    """Unified heading detection across all patterns.

    Returns:
        Dict with keys: 'is_heading', 'title', 'level', 'type'
        where type is one of: 'book', 'chapter', 'ordinal', 'roman', 'caps',
                              'prologue', 'markdown', 'separator'
    """
    stripped = line.strip()

    if not stripped or is_ocr_artifact(stripped):
        return {'is_heading': False, 'title': '', 'level': 0, 'type': None}

    # Check in order of specificity (most specific first)

    # Markdown headers (explicit structure)
    is_md, md_level, md_title = detect_markdown_header(stripped)
    if is_md:
        return {'is_heading': True, 'title': md_title, 'level': md_level, 'type': 'markdown'}

    # Book/Part headings (highest structural level)
    is_book, book_title, book_level = detect_book_heading(stripped)
    if is_book:
        return {'is_heading': True, 'title': book_title, 'level': 1, 'type': 'book'}

    # Chapter headings
    is_chapter, chapter_title, chapter_level = detect_chapter_heading(stripped)
    if is_chapter:
        return {'is_heading': True, 'title': chapter_title, 'level': 2, 'type': 'chapter'}

    # Ordinal headings (PREMIER MIRACLE, etc.)
    is_ordinal, ordinal_title, ordinal_level = detect_ordinal_heading(stripped)
    if is_ordinal:
        return {'is_heading': True, 'title': ordinal_title, 'level': 2, 'type': 'ordinal'}

    # Prologue/Preface headings
    is_prologue, prologue_title, prologue_level = detect_prologue_heading(stripped)
    if is_prologue:
        return {'is_heading': True, 'title': prologue_title, 'level': 2, 'type': 'prologue'}

    # Roman numeral sections
    is_roman, roman_title = detect_roman_numeral_heading(stripped)
    if is_roman:
        return {'is_heading': True, 'title': roman_title, 'level': 3, 'type': 'roman'}

    # ALL CAPS headings (general catch-all)
    if is_all_caps_heading(stripped):
        # Determine level based on length
        level = 2 if len(stripped) > 40 else 3
        return {'is_heading': True, 'title': stripped, 'level': level, 'type': 'caps'}

    return {'is_heading': False, 'title': '', 'level': 0, 'type': None}


def extract_nodes_from_txt(txt_content: str) -> tuple[list[dict], list[str]]:
    """Extract heading nodes from plain text content.

    Returns:
        Tuple of (node_list, lines) where node_list contains
        {'node_title': str, 'line_num': int, 'level': int, 'type': str}
    """
    lines = txt_content.split('\n')
    node_list = []

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Handle separators (they indicate structure but aren't headings themselves)
        if detect_separator(stripped):
            # Look at the next non-empty line for a potential heading
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines):
                next_heading = detect_heading(lines[j])
                if next_heading['is_heading']:
                    # Skip the separator, process the heading
                    i = j
                    continue
            i += 1
            continue

        heading = detect_heading(stripped)

        if heading['is_heading']:
            # Check for multi-line heading (OCR artifact - continued ALL CAPS)
            merged_title = heading['title']
            j = i + 1

            # Only merge if current is ALL CAPS type
            if heading['type'] == 'caps':
                while j < len(lines):
                    next_stripped = lines[j].strip()

                    # Skip empty lines (allow one)
                    if not next_stripped:
                        j += 1
                        if j < len(lines) and not lines[j].strip():
                            break  # Two empty lines = stop merging
                        continue

                    # If next line is also short ALL CAPS, merge
                    if is_all_caps_heading(next_stripped) and len(next_stripped) < 40:
                        merged_title = merged_title.rstrip() + " " + next_stripped
                        j += 1
                    else:
                        break
            else:
                j = i + 1  # No merging for non-caps headings

            node_list.append({
                'node_title': merged_title,
                'line_num': i + 1,  # 1-indexed
                'level': heading['level'],
                'type': heading['type'],
            })
            i = j
        else:
            i += 1

    return node_list, lines


def extract_node_text_content(node_list: list[dict], txt_lines: list[str]) -> list[dict]:
    """Extract text content for each node (from heading to next heading)."""
    result = []

    for i, node in enumerate(node_list):
        start_line = node['line_num'] - 1  # 0-indexed

        if i + 1 < len(node_list):
            end_line = node_list[i + 1]['line_num'] - 1
        else:
            end_line = len(txt_lines)

        # Extract and clean text
        text_lines = []
        for line in txt_lines[start_line:end_line]:
            stripped = line.strip()
            if stripped and not is_ocr_artifact(stripped):
                text_lines.append(stripped)

        result.append({
            'title': node['node_title'],
            'line_num': node['line_num'],
            'level': node['level'],
            'type': node.get('type', 'unknown'),
            'text': '\n'.join(text_lines),
        })

    return result


def build_tree_from_nodes(node_list: list[dict]) -> list[dict]:
    """Build hierarchical tree from flat node list based on levels."""
    if not node_list:
        return []

    stack = []
    root_nodes = []
    node_counter = 0

    for node in node_list:
        current_level = node['level']

        tree_node = {
            'title': node['title'],
            'node_id': str(node_counter).zfill(4),
            'text': node['text'],
            'line_num': node['line_num'],
            'type': node.get('type', 'unknown'),
            'nodes': [],
        }
        node_counter += 1

        # Pop stack until we find a parent with lower level
        while stack and stack[-1][1] >= current_level:
            stack.pop()

        if not stack:
            root_nodes.append(tree_node)
        else:
            parent_node, _ = stack[-1]
            parent_node['nodes'].append(tree_node)

        stack.append((tree_node, current_level))

    return root_nodes


async def txt_to_tree(
    txt_path: str,
    if_add_node_summary: str = 'no',
    summary_token_threshold: int = 200,
    model: str = None,
    if_add_doc_description: str = 'no',
    if_add_node_text: str = 'no',
    if_add_node_id: str = 'yes',
) -> dict:
    """Convert a plain text file to a PageIndex tree structure.

    Args:
        txt_path: Path to the .txt file
        if_add_node_summary: Whether to generate LLM summaries for nodes
        summary_token_threshold: Token threshold for summary generation
        model: LLM model to use for summaries
        if_add_doc_description: Whether to generate document description
        if_add_node_text: Whether to include full text in output
        if_add_node_id: Whether to add node IDs

    Returns:
        Dict with 'doc_name' and 'structure' keys
    """
    with open(txt_path, encoding='utf-8', errors='replace') as f:
        txt_content = f.read()

    print("Extracting nodes from text...")
    node_list, txt_lines = extract_nodes_from_txt(txt_content)

    print(f"Found {len(node_list)} heading nodes")

    # Show heading type distribution
    type_counts = {}
    for node in node_list:
        t = node.get('type', 'unknown')
        type_counts[t] = type_counts.get(t, 0) + 1
    if type_counts:
        print("  Heading types: " + ", ".join(f"{k}={v}" for k, v in sorted(type_counts.items())))

    print("Extracting text content for each node...")
    nodes_with_content = extract_node_text_content(node_list, txt_lines)

    print("Building tree from nodes...")
    tree_structure = build_tree_from_nodes(nodes_with_content)

    if if_add_node_id == 'yes':
        write_node_id(tree_structure)

    print("Formatting tree structure...")

    if if_add_node_summary == 'yes':
        tree_structure = format_structure(
            tree_structure,
            order=['title', 'node_id', 'summary', 'prefix_summary', 'text', 'line_num', 'nodes']
        )

        print("Generating summaries for each node...")
        tree_structure = await generate_summaries_for_structure_md(
            tree_structure,
            summary_token_threshold=summary_token_threshold,
            model=model
        )

        if if_add_node_text == 'no':
            tree_structure = format_structure(
                tree_structure,
                order=['title', 'node_id', 'summary', 'prefix_summary', 'line_num', 'nodes']
            )

        if if_add_doc_description == 'yes':
            print("Generating document description...")
            clean_structure = create_clean_structure_for_description(tree_structure)
            doc_description = generate_doc_description(clean_structure, model=model)
            return {
                'doc_name': os.path.splitext(os.path.basename(txt_path))[0],
                'doc_description': doc_description,
                'structure': tree_structure,
            }
    else:
        if if_add_node_text == 'yes':
            tree_structure = format_structure(
                tree_structure,
                order=['title', 'node_id', 'text', 'line_num', 'nodes']
            )
        else:
            tree_structure = format_structure(
                tree_structure,
                order=['title', 'node_id', 'line_num', 'nodes']
            )

    return {
        'doc_name': os.path.splitext(os.path.basename(txt_path))[0],
        'structure': tree_structure,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python page_index_txt.py <txt_path>")
        sys.exit(1)

    txt_path = sys.argv[1]

    result = asyncio.run(txt_to_tree(
        txt_path=txt_path,
        if_add_node_summary='no',
        if_add_node_text='yes',
    ))

    print('\n' + '=' * 60)
    print('TREE STRUCTURE')
    print('=' * 60)

    # Print simplified tree
    def print_tree(nodes, indent=0):
        for node in nodes:
            title = node['title'][:50] + '...' if len(node['title']) > 50 else node['title']
            node_type = node.get('type', '?')
            print('  ' * indent + f"[{node.get('node_id', '?')}] ({node_type}) {title}")
            if node.get('nodes'):
                print_tree(node['nodes'], indent + 1)

    print_tree(result['structure'])

    # Save to file
    output_path = f"./results/{result['doc_name']}_structure.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nTree structure saved to: {output_path}")
