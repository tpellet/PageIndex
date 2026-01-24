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
import logging
import os
import re
from typing import Optional

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
    # Fallback for direct script execution and tests
    from page_index_md import generate_summaries_for_structure_md  # type: ignore[no-redef]
    from utils import (  # type: ignore[no-redef]
        count_tokens,
        create_clean_structure_for_description,
        format_structure,
        generate_doc_description,
        write_node_id,
    )

logger = logging.getLogger(__name__)

# Maximum length for structural headings (book/part markers, ordinals)
# Lines longer than this are assumed to be prose, not standalone headings
MAX_STRUCTURAL_HEADING_LENGTH = 60


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
    # Translator/editor metadata
    "traduction complète en français moderne",
    "langue originale",
    "date de traduction",
    "notes du traducteur",
    "glossaire des termes",
}

# Patterns for OCR noise
OCR_NOISE_PATTERNS = [
    r'^\d+\s*$',  # Standalone page numbers
    r'^[\^\\\|/\*\-\.\,\;\:\'\"\s]+$',  # Punctuation-only lines
    r'^[a-zA-Z]\s*$',  # Single letters
    r'^\s*\d+\s*,\s*$',  # "123," patterns
    r'^http[s]?://',  # URLs
    r'^ark:/',  # Archive.org identifiers
    r'^\*\s+',  # Footnote markers
    r'^source\s*:\s*',  # Source metadata (case-insensitive via re.IGNORECASE)
]


def is_metadata_line(line: str) -> bool:
    """Detect metadata key-value pairs like 'Source: ...' or 'Date: ...'."""
    stripped = line.strip()
    # Common metadata patterns: "Key: value" at start of line
    metadata_keys = [
        'source', 'date', 'translator', 'traducteur', 'author', 'auteur',
        'editor', 'éditeur', 'publisher', 'éditeur', 'edition', 'édition',
    ]
    lower = stripped.lower()
    for key in metadata_keys:
        if lower.startswith(f'{key}:') or lower.startswith(f'{key} :'):
            return True
    return False


def is_ocr_artifact(line: str) -> bool:
    """Check if line is an OCR artifact or noise."""
    stripped = line.strip().lower()
    if not stripped:
        return False

    # Direct matches
    if stripped in OCR_ARTIFACTS:
        return True

    # Metadata lines (Source: ..., Date: ..., etc.)
    if is_metadata_line(line):
        return True

    # Pattern matches
    for pattern in OCR_NOISE_PATTERNS:
        if re.match(pattern, stripped, re.IGNORECASE):
            return True

    # Very short gibberish (OCR errors)
    if len(stripped) < 4 and not stripped.isalpha():
        return True

    return False


def is_all_caps_heading(line: str) -> bool:
    """Check if line is an all-caps heading.

    Supports French accented capitals and common punctuation including
    French guillemets («») and curly quotes.
    """
    stripped = line.strip()
    if len(stripped) < 5:
        return False

    # Allow uppercase letters (including French accented), spaces, hyphens, common punctuation
    # Includes «»''""  for French quotation marks and curly quotes
    # Pattern: starts with uppercase or opening quote, contains mostly uppercase
    if re.match(r'^[A-ZÀÂÄÉÈÊËÏÎÔÙÛÜÇ«\'""][A-ZÀÂÄÉÈÊËÏÎÔÙÛÜÇ\s\-,\.\'\"()«»''""]+$', stripped):
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
    if len(stripped) > MAX_STRUCTURAL_HEADING_LENGTH:
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
    if len(stripped) > MAX_STRUCTURAL_HEADING_LENGTH:
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

    stack: list[tuple[dict, int]] = []
    root_nodes: list[dict] = []
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


def txt_to_page_list(
    txt_content: str,
    chars_per_page: int = 3000,
    model: str = "gpt-4o-2024-11-20"
) -> tuple[list[tuple[str, int]], list[int]]:
    """Create synthetic page list from text content.

    Splits text on paragraph boundaries (double newlines) and accumulates
    paragraphs until chars_per_page threshold is reached. This enables
    PageIndex RAG functions to work with txt files.

    Args:
        txt_content: The full text content
        chars_per_page: Target characters per synthetic page (default 3000)
        model: Model for token counting (default gpt-4o-2024-11-20)

    Returns:
        Tuple of:
        - page_list: List of (text_chunk, token_count) tuples
        - line_to_page: List mapping line number (0-indexed) to page (1-indexed)
    """
    if not txt_content:
        return [], []

    lines = txt_content.split('\n')
    paragraphs = txt_content.split('\n\n')

    # Build page list by accumulating paragraphs
    page_list: list[tuple[str, int]] = []
    current_page_text: list[str] = []
    current_char_count = 0

    for para in paragraphs:
        para_stripped = para.strip()
        if not para_stripped:
            continue

        para_len = len(para_stripped)

        # Start new page if adding this paragraph exceeds threshold
        if current_char_count > 0 and current_char_count + para_len > chars_per_page:
            page_text = '\n\n'.join(current_page_text)
            token_count = count_tokens(page_text, model=model)
            page_list.append((page_text, token_count))
            current_page_text = []
            current_char_count = 0

        current_page_text.append(para_stripped)
        current_char_count += para_len

    # Don't forget the last page
    if current_page_text:
        page_text = '\n\n'.join(current_page_text)
        token_count = count_tokens(page_text, model=model)
        page_list.append((page_text, token_count))

    # Build line-to-page mapping
    # For each line, find which page it belongs to by position
    line_to_page = []
    char_pos = 0
    page_boundaries = []  # Character positions where each page starts

    # Calculate page boundaries
    boundary = 0
    for page_text, _ in page_list:
        page_boundaries.append(boundary)
        # Account for paragraph separators we stripped/rejoined
        boundary += len(page_text) + 2  # +2 for \n\n separator

    # Map each line to its page
    for line in lines:
        # Find which page this character position falls into
        page_num = 1
        for i, page_boundary in enumerate(page_boundaries):
            if i + 1 < len(page_boundaries):
                if page_boundary <= char_pos < page_boundaries[i + 1]:
                    page_num = i + 1
                    break
            else:
                page_num = len(page_boundaries)

        line_to_page.append(page_num)
        char_pos += len(line) + 1  # +1 for newline

    return page_list, line_to_page


def add_page_indices_to_nodes(
    tree_structure: list[dict],
    line_to_page: list[int],
    total_pages: int
) -> list[dict]:
    """Add physical_index, start_index, end_index to nodes.

    These indices enable PageIndex RAG functions like check_title_appearance
    and add_node_text to work with txt files.

    Args:
        tree_structure: The tree structure from build_tree_from_nodes
        line_to_page: Mapping from line number (0-indexed) to page (1-indexed)
        total_pages: Total number of synthetic pages

    Returns:
        The tree structure with page indices added to each node
    """
    def process_nodes(nodes: list[dict], parent_next_line: Optional[int] = None) -> None:
        for i, node in enumerate(nodes):
            line_num = node.get('line_num', 1)

            # Determine physical_index from line number
            if line_num > 0 and line_num <= len(line_to_page):
                physical_index = line_to_page[line_num - 1]  # Convert to 0-indexed
            else:
                physical_index = 1

            node['physical_index'] = physical_index
            node['start_index'] = physical_index

            # Determine end_index
            # Look for next sibling or parent's next sibling
            if i + 1 < len(nodes):
                next_line = nodes[i + 1].get('line_num', 1)
                if next_line > 0 and next_line <= len(line_to_page):
                    end_page = line_to_page[next_line - 1] - 1
                    node['end_index'] = max(physical_index, end_page)
                else:
                    node['end_index'] = total_pages
            elif parent_next_line is not None:
                if parent_next_line > 0 and parent_next_line <= len(line_to_page):
                    end_page = line_to_page[parent_next_line - 1] - 1
                    node['end_index'] = max(physical_index, end_page)
                else:
                    node['end_index'] = total_pages
            else:
                node['end_index'] = total_pages

            # Process children
            if node.get('nodes'):
                # Child nodes should end before the next sibling of current node
                if i + 1 < len(nodes):
                    child_parent_next = nodes[i + 1].get('line_num')
                elif parent_next_line:
                    child_parent_next = parent_next_line
                else:
                    child_parent_next = None
                process_nodes(node['nodes'], child_parent_next)

    process_nodes(tree_structure)
    return tree_structure


async def txt_to_tree(
    txt_path: str,
    if_add_node_summary: str = 'no',
    summary_token_threshold: int = 200,
    model: Optional[str] = None,
    if_add_doc_description: str = 'no',
    if_add_node_text: str = 'no',
    if_add_node_id: str = 'yes',
    chars_per_page: int = 3000,
) -> dict:
    """Convert a plain text file to a PageIndex tree structure.

    Args:
        txt_path: Path to the .txt file
        if_add_node_summary: Whether to generate LLM summaries for nodes
        summary_token_threshold: Token threshold for summary generation
        model: LLM model to use for summaries
        if_add_doc_description: Whether to generate document description
        if_add_node_text: Whether to include full text in output. When 'yes',
            also generates synthetic pages and adds page indices to nodes.
        if_add_node_id: Whether to add node IDs
        chars_per_page: Target characters per synthetic page (default 3000).
            Only used when if_add_node_text='yes'.

    Returns:
        Dict with 'doc_name' and 'structure' keys. When if_add_node_text='yes',
        also includes 'page_list' (list of (text, token_count) tuples) for
        PageIndex RAG compatibility.
    """
    with open(txt_path, encoding='utf-8', errors='replace') as f:
        txt_content = f.read()

    logger.info("Extracting nodes from text...")
    node_list, txt_lines = extract_nodes_from_txt(txt_content)

    logger.info(f"Found {len(node_list)} heading nodes")

    # Show heading type distribution
    type_counts: dict[str, int] = {}
    for node in node_list:
        t = node.get('type', 'unknown')
        type_counts[t] = type_counts.get(t, 0) + 1
    if type_counts:
        logger.info("  Heading types: " + ", ".join(f"{k}={v}" for k, v in sorted(type_counts.items())))

    logger.info("Extracting text content for each node...")
    nodes_with_content = extract_node_text_content(node_list, txt_lines)

    logger.info("Building tree from nodes...")
    tree_structure = build_tree_from_nodes(nodes_with_content)

    # Generate synthetic pages and add page indices when including node text
    page_list = None
    if if_add_node_text == 'yes':
        logger.info("Generating synthetic pages...")
        page_list, line_to_page = txt_to_page_list(
            txt_content,
            chars_per_page=chars_per_page,
            model=model or "gpt-4o-2024-11-20"
        )
        total_pages = len(page_list)
        logger.info(f"Created {total_pages} synthetic pages")

        if line_to_page:
            logger.info("Adding page indices to nodes...")
            tree_structure = add_page_indices_to_nodes(tree_structure, line_to_page, total_pages)

    if if_add_node_id == 'yes':
        write_node_id(tree_structure)

    logger.info("Formatting tree structure...")

    if if_add_node_summary == 'yes':
        tree_structure = format_structure(
            tree_structure,
            order=['title', 'node_id', 'summary', 'prefix_summary', 'text', 'line_num', 'nodes']
        )

        logger.info("Generating summaries for each node...")
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
            logger.info("Generating document description...")
            clean_structure = create_clean_structure_for_description(tree_structure)
            doc_description = generate_doc_description(clean_structure, model=model)
            result = {
                'doc_name': os.path.splitext(os.path.basename(txt_path))[0],
                'doc_description': doc_description,
                'structure': tree_structure,
            }
            if page_list is not None:
                result['page_list'] = page_list
            return result
    else:
        if if_add_node_text == 'yes':
            tree_structure = format_structure(
                tree_structure,
                order=['title', 'node_id', 'text', 'line_num', 'physical_index', 'start_index', 'end_index', 'nodes']
            )
        else:
            tree_structure = format_structure(
                tree_structure,
                order=['title', 'node_id', 'line_num', 'nodes']
            )

    result = {
        'doc_name': os.path.splitext(os.path.basename(txt_path))[0],
        'structure': tree_structure,
    }
    if page_list is not None:
        result['page_list'] = page_list
    return result


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
