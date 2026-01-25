"""Tests for PageIndex txt file support."""

import sys
from pathlib import Path

import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'pageindex'))

from page_index_txt import (
    add_page_indices_to_nodes,
    build_tree_from_nodes,
    detect_book_heading,
    detect_chapter_heading,
    detect_heading,
    detect_markdown_header,
    detect_ordinal_heading,
    detect_prologue_heading,
    detect_roman_numeral_heading,
    detect_separator,
    extract_node_text_content,
    extract_nodes_from_txt,
    is_all_caps_heading,
    is_metadata_line,
    is_ocr_artifact,
    txt_to_page_list,
)


class TestOCRArtifactDetection:
    """Tests for OCR artifact filtering."""

    def test_google_digitized(self):
        assert is_ocr_artifact("Digitized by Google") is True
        assert is_ocr_artifact("digitized by google") is True
        assert is_ocr_artifact("DIGITIZED BY GOOGLE") is True

    def test_page_numbers(self):
        assert is_ocr_artifact("123") is True
        assert is_ocr_artifact("  456  ") is True
        assert is_ocr_artifact("12") is True

    def test_punctuation_only(self):
        assert is_ocr_artifact("---") is True
        assert is_ocr_artifact("...") is True
        assert is_ocr_artifact("^^^") is True

    def test_normal_text_not_artifact(self):
        assert is_ocr_artifact("This is normal text") is False
        assert is_ocr_artifact("CHAPTER ONE") is False
        assert is_ocr_artifact("Louis IX of France") is False

    def test_urls(self):
        assert is_ocr_artifact("http://example.com") is True
        assert is_ocr_artifact("https://books.google.com") is True

    def test_copyright(self):
        assert is_ocr_artifact("Copyright") is True
        assert is_ocr_artifact("All rights reserved") is True


class TestAllCapsHeading:
    """Tests for ALL CAPS heading detection."""

    def test_basic_caps(self):
        assert is_all_caps_heading("CHAPTER ONE") is True
        assert is_all_caps_heading("THE CRUSADE OF ST. LEWIS") is True

    def test_french_accented_caps(self):
        assert is_all_caps_heading("PREMIÈRE PARTIE") is True
        assert is_all_caps_heading("L'ÉGLISE DE FRANCE") is True

    def test_too_short(self):
        assert is_all_caps_heading("THE") is False
        assert is_all_caps_heading("OF") is False

    def test_mixed_case_not_caps(self):
        assert is_all_caps_heading("Chapter One") is False
        assert is_all_caps_heading("The CRUSADE") is False

    def test_caps_with_punctuation(self):
        assert is_all_caps_heading("ST. LEWIS") is True
        assert is_all_caps_heading("SAINT-DENIS") is True


class TestMarkdownHeader:
    """Tests for markdown header detection."""

    def test_h1(self):
        is_header, level, title = detect_markdown_header("# Title")
        assert is_header is True
        assert level == 1
        assert title == "Title"

    def test_h2(self):
        is_header, level, title = detect_markdown_header("## Section")
        assert is_header is True
        assert level == 2
        assert title == "Section"

    def test_h3_with_spaces(self):
        is_header, level, title = detect_markdown_header("###   Subsection Name  ")
        assert is_header is True
        assert level == 3
        assert title == "Subsection Name"

    def test_not_markdown(self):
        is_header, level, title = detect_markdown_header("Regular text")
        assert is_header is False

    def test_hash_without_space(self):
        is_header, level, title = detect_markdown_header("#NoSpace")
        assert is_header is False


class TestSeparator:
    """Tests for separator detection."""

    def test_dashes(self):
        assert detect_separator("---") is True
        assert detect_separator("------") is True

    def test_equals(self):
        assert detect_separator("===") is True
        assert detect_separator("==========") is True

    def test_not_separator(self):
        assert detect_separator("--") is False
        assert detect_separator("text---text") is False


class TestRomanNumeralHeading:
    """Tests for Roman numeral section headings."""

    def test_simple_roman(self):
        is_heading, title = detect_roman_numeral_heading("I. Introduction")
        assert is_heading is True
        assert title == "I. Introduction"

    def test_large_roman(self):
        is_heading, title = detect_roman_numeral_heading("XII. De la vie du roi")
        assert is_heading is True

    def test_roman_with_paren(self):
        is_heading, title = detect_roman_numeral_heading("IV) Section Four")
        assert is_heading is True

    def test_not_roman_heading(self):
        is_heading, _ = detect_roman_numeral_heading("In the beginning")
        assert is_heading is False


class TestChapterHeading:
    """Tests for chapter heading detection."""

    def test_english_chapter(self):
        is_heading, title, level = detect_chapter_heading("CHAPTER I")
        assert is_heading is True
        assert level == 1

    def test_french_chapter(self):
        is_heading, title, level = detect_chapter_heading("CHAPITRE II")
        assert is_heading is True

    def test_chapter_with_roman(self):
        is_heading, _, _ = detect_chapter_heading("Chapter XII")
        assert is_heading is True

    def test_chapter_with_number(self):
        is_heading, _, _ = detect_chapter_heading("CHAPTER 3")
        assert is_heading is True

    def test_not_chapter(self):
        is_heading, _, _ = detect_chapter_heading("This chapter discusses")
        assert is_heading is False


class TestBookHeading:
    """Tests for book/part heading detection."""

    def test_first_book(self):
        is_heading, title, level = detect_book_heading("FIRST BOOK")
        assert is_heading is True
        assert level == 1

    def test_second_part(self):
        is_heading, _, _ = detect_book_heading("SECOND PART")
        assert is_heading is True

    def test_french_livre(self):
        is_heading, _, _ = detect_book_heading("LIVRE PREMIER")
        assert is_heading is True

    def test_book_with_roman(self):
        is_heading, _, _ = detect_book_heading("BOOK III")
        assert is_heading is True

    def test_false_positive_prevention(self):
        """Ensure we don't match sentences starting with ordinals."""
        is_heading, _, _ = detect_book_heading("premier s'appelait Henri, le second Thibaut")
        assert is_heading is False

    def test_too_long_not_book(self):
        long_title = "FIRST BOOK OF THE CHRONICLES OF THE KINGS OF FRANCE AND THEIR DEEDS"
        is_heading, _, _ = detect_book_heading(long_title)
        assert is_heading is False


class TestOrdinalHeading:
    """Tests for ordinal heading detection (PREMIER MIRACLE, etc.)."""

    def test_premier_miracle(self):
        is_heading, title, level = detect_ordinal_heading("PREMIER MIRACLE")
        assert is_heading is True
        assert level == 2

    def test_deuxieme_chapitre(self):
        is_heading, _, _ = detect_ordinal_heading("DEUXIÈME CHAPITRE")
        assert is_heading is True

    def test_tenth(self):
        is_heading, _, _ = detect_ordinal_heading("DIXIÈME MIRACLE")
        assert is_heading is True

    def test_not_ordinal(self):
        is_heading, _, _ = detect_ordinal_heading("Le premier jour")
        assert is_heading is False


class TestPrologueHeading:
    """Tests for prologue/preface heading detection."""

    def test_prologue(self):
        is_heading, title, level = detect_prologue_heading("PROLOGUE")
        assert is_heading is True

    def test_preface(self):
        is_heading, _, _ = detect_prologue_heading("PRÉFACE")
        assert is_heading is True

    def test_introduction(self):
        is_heading, _, _ = detect_prologue_heading("INTRODUCTION")
        assert is_heading is True

    def test_avertissement(self):
        is_heading, _, _ = detect_prologue_heading("AVERTISSEMENT DE L'ÉDITEUR")
        assert is_heading is True


class TestUnifiedHeadingDetection:
    """Tests for the unified detect_heading function."""

    def test_markdown_priority(self):
        result = detect_heading("# Chapter Title")
        assert result['is_heading'] is True
        assert result['type'] == 'markdown'

    def test_book_priority(self):
        result = detect_heading("FIRST BOOK")
        assert result['is_heading'] is True
        assert result['type'] == 'book'

    def test_chapter_detected(self):
        result = detect_heading("CHAPTER III")
        assert result['is_heading'] is True
        assert result['type'] == 'chapter'

    def test_roman_detected(self):
        result = detect_heading("IV. De la vie du roi")
        assert result['is_heading'] is True
        assert result['type'] == 'roman'

    def test_caps_fallback(self):
        result = detect_heading("THE CRUSADE OF ST. LEWIS")
        assert result['is_heading'] is True
        assert result['type'] == 'caps'

    def test_ocr_artifact_filtered(self):
        result = detect_heading("Digitized by Google")
        assert result['is_heading'] is False

    def test_normal_text_not_heading(self):
        result = detect_heading("This is normal prose text that tells a story.")
        assert result['is_heading'] is False


class TestMultiLineHeadingMerging:
    """Tests for multi-line ALL CAPS heading detection (OCR artifact handling)."""

    def test_merge_two_caps_lines(self):
        """Test that consecutive short ALL CAPS lines are merged."""
        content = """FIRST BOOK
OF THE

Life of the king."""

        nodes, _ = extract_nodes_from_txt(content)

        # "FIRST BOOK" and "OF THE" should merge
        assert len(nodes) >= 1
        assert "FIRST BOOK" in nodes[0]['node_title']

    def test_stop_merging_on_double_empty_line(self):
        """Test that two consecutive empty lines stop merging."""
        content = """FIRST HEADING


SECOND HEADING

Content text."""

        nodes, _ = extract_nodes_from_txt(content)

        # Should have two separate headings
        assert len(nodes) == 2
        assert nodes[0]['node_title'] == "FIRST HEADING"
        assert nodes[1]['node_title'] == "SECOND HEADING"

    def test_separator_followed_by_no_heading(self):
        """Test separator followed by non-heading content."""
        content = """Some intro text.

---

Regular paragraph text here.

CHAPTER ONE

Chapter content."""

        nodes, _ = extract_nodes_from_txt(content)

        # Should only find CHAPTER ONE as heading
        assert any('CHAPTER ONE' in n['node_title'] for n in nodes)


class TestNodeExtraction:
    """Tests for extracting nodes from text content."""

    def test_simple_extraction(self):
        content = """FIRST BOOK

This is the content of the first book.
It has multiple lines.

SECOND BOOK

This is the second book content."""

        nodes, lines = extract_nodes_from_txt(content)

        assert len(nodes) == 2
        assert nodes[0]['node_title'] == "FIRST BOOK"
        assert nodes[0]['type'] == 'book'
        assert nodes[1]['node_title'] == "SECOND BOOK"

    def test_mixed_heading_types(self):
        content = """# Document Title

PROLOGUE

Introduction text.

I. First Section

Content of first section.

II. Second Section

Content of second section."""

        nodes, lines = extract_nodes_from_txt(content)

        assert len(nodes) == 4
        assert nodes[0]['type'] == 'markdown'
        assert nodes[1]['type'] == 'prologue'
        assert nodes[2]['type'] == 'roman'
        assert nodes[3]['type'] == 'roman'

    def test_ocr_artifacts_filtered(self):
        content = """CHAPTER ONE

Digitized by Google

Real content here.

123

More content.

CHAPTER TWO

Second chapter content."""

        nodes, lines = extract_nodes_from_txt(content)

        # Should only have 2 chapters, not the OCR artifacts
        assert len(nodes) == 2
        assert nodes[0]['node_title'] == "CHAPTER ONE"
        assert nodes[1]['node_title'] == "CHAPTER TWO"


class TestTreeBuilding:
    """Tests for building hierarchical tree from nodes."""

    def test_flat_structure(self):
        nodes = [
            {'title': 'A', 'text': 'a', 'level': 1, 'line_num': 1},
            {'title': 'B', 'text': 'b', 'level': 1, 'line_num': 10},
            {'title': 'C', 'text': 'c', 'level': 1, 'line_num': 20},
        ]

        tree = build_tree_from_nodes(nodes)

        assert len(tree) == 3
        assert all(len(node['nodes']) == 0 for node in tree)

    def test_nested_structure(self):
        nodes = [
            {'title': 'Book 1', 'text': '', 'level': 1, 'line_num': 1},
            {'title': 'Chapter 1', 'text': '', 'level': 2, 'line_num': 10},
            {'title': 'Section 1.1', 'text': '', 'level': 3, 'line_num': 20},
            {'title': 'Chapter 2', 'text': '', 'level': 2, 'line_num': 30},
            {'title': 'Book 2', 'text': '', 'level': 1, 'line_num': 40},
        ]

        tree = build_tree_from_nodes(nodes)

        assert len(tree) == 2  # Two books at root
        assert tree[0]['title'] == 'Book 1'
        assert len(tree[0]['nodes']) == 2  # Two chapters under Book 1
        assert len(tree[0]['nodes'][0]['nodes']) == 1  # One section under Chapter 1

    def test_empty_input(self):
        tree = build_tree_from_nodes([])
        assert tree == []


class TestIntegration:
    """Integration tests with real-world-like content."""

    def test_joinville_style(self):
        """Test content similar to Joinville chronicle structure."""
        content = """JOINVILLE'S CHRONICLE OF THE CRUSADE OF ST. LEWIS

DEDICATION AND DIVISION OF THE WORK

To his good lord Lewis, son of the King of France, by the
grace of God King of Navarre...

FIRST BOOK

OF ST. LEWIS

The first book tells of the virtuous life of the king.

REGARD OF ST. LEWIS FOR WORTH AND UPRIGHTNESS

The holy king loved all persons who devoted themselves to
the service of God...

SECOND BOOK

BIRTH AND CORONATION OF ST. LEWIS

In the year of grace 1215, on St. Mark's day..."""

        nodes, lines = extract_nodes_from_txt(content)
        tree = build_tree_from_nodes([
            {'title': n['node_title'], 'text': '', 'level': n['level'], 'line_num': n['line_num']}
            for n in nodes
        ])

        # Should detect multiple headings
        assert len(nodes) >= 5

        # FIRST BOOK and SECOND BOOK should be at book level
        book_nodes = [n for n in nodes if n['type'] == 'book']
        assert len(book_nodes) == 2

        # Tree should have book-level nodes at root
        assert len(tree) >= 2

    def test_miracles_style(self):
        """Test content similar to Miracles de Saint Louis structure."""
        content = """MIRACLES DE SAINT LOUIS

PAR LE CONFESSEUR DE LA REINE MARGUERITE

---

PREMIER MIRACLE

Marote, la fille de Fressent d'Arras...

---

DEUXIÈME MIRACLE

D'une femme qui fut guérie au tombeau de saint Louis...

---

TROISIÈME MIRACLE

Ce miracle est d'une femme qui avait perdu le corps..."""

        nodes, lines = extract_nodes_from_txt(content)

        # Should detect ordinal headings
        ordinal_nodes = [n for n in nodes if n['type'] == 'ordinal']
        assert len(ordinal_nodes) == 3

    def test_geoffrey_beaulieu_style(self):
        """Test content with Roman numeral sections."""
        content = """# Geoffroy de Beaulieu - Vie et sainte conduite de Louis

PROLOGUE

Pour la gloire et l'honneur du nom divin...

I. Comment l'éloge du roi Josias convient au roi Louis

En premier lieu donc, pour recommander le pieux roi...

II. Que le nom de Josias lui convient

Il suffit de dire pour l'instant comment le sens du nom...

III. De l'innocence de sa vie et de sa sainte conduite

Assurément, on peut dire de lui..."""

        nodes, lines = extract_nodes_from_txt(content)

        # Should detect markdown, prologue, and roman
        assert any(n['type'] == 'markdown' for n in nodes)
        assert any(n['type'] == 'prologue' for n in nodes)
        assert any(n['type'] == 'roman' for n in nodes)

        roman_nodes = [n for n in nodes if n['type'] == 'roman']
        assert len(roman_nodes) == 3


class TestFrenchQuotationMarks:
    """Tests for French quotation marks in ALL CAPS detection."""

    def test_guillemets_in_caps(self):
        """Test that guillemets (« ») are allowed in ALL CAPS headings."""
        assert is_all_caps_heading("LE ROI DIT «OUI»") is True
        assert is_all_caps_heading("«PREMIÈRE PARTIE»") is True
        assert is_all_caps_heading("CHAPITRE «LE DÉPART»") is True

    def test_curly_quotes_in_caps(self):
        """Test that curly quotes are allowed in ALL CAPS headings."""
        assert is_all_caps_heading('LE ROI DIT "OUI"') is True
        assert is_all_caps_heading("'PREMIÈRE PARTIE'") is True
        assert is_all_caps_heading("CHAPITRE 'LE DÉPART'") is True

    def test_mixed_quotes_in_caps(self):
        """Test mixed quote styles in ALL CAPS headings."""
        assert is_all_caps_heading("LE «ROI» DIT 'OUI'") is True


class TestOCRArtifactsExpanded:
    """Tests for expanded OCR artifact filtering."""

    def test_translator_metadata(self):
        """Test that translator metadata is detected as artifact."""
        assert is_ocr_artifact("traduction complète en français moderne") is True
        assert is_ocr_artifact("langue originale") is True
        assert is_ocr_artifact("date de traduction") is True
        assert is_ocr_artifact("notes du traducteur") is True
        assert is_ocr_artifact("glossaire des termes") is True

    def test_footnote_markers(self):
        """Test that footnote markers are detected as noise."""
        assert is_ocr_artifact("* See appendix A") is True
        assert is_ocr_artifact("*  Reference to chapter 3") is True

    def test_source_metadata(self):
        """Test that source metadata lines are detected."""
        assert is_metadata_line("Source: Archive.org") is True
        assert is_metadata_line("source: Google Books") is True
        assert is_metadata_line("Source : Bibliothèque nationale") is True
        assert is_metadata_line("Date: 1856") is True
        assert is_metadata_line("Translator: John Smith") is True
        assert is_metadata_line("Traducteur: Jean Dupont") is True

    def test_normal_text_not_metadata(self):
        """Test that normal text is not detected as metadata."""
        assert is_metadata_line("The source of all wisdom") is False
        assert is_metadata_line("Dating back to the crusades") is False


class TestSyntheticPages:
    """Tests for synthetic page generation."""

    def test_txt_to_page_list_basic(self):
        """Test basic page list generation."""
        content = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        page_list, line_to_page = txt_to_page_list(content, chars_per_page=100)

        assert len(page_list) >= 1
        assert all(isinstance(p, tuple) and len(p) == 2 for p in page_list)
        assert all(isinstance(p[0], str) and isinstance(p[1], int) for p in page_list)

    def test_paragraph_boundary_breaks(self):
        """Test that pages break on paragraph boundaries."""
        # Create content with clear paragraphs
        para1 = "A" * 50
        para2 = "B" * 50
        para3 = "C" * 50
        content = f"{para1}\n\n{para2}\n\n{para3}"

        # Set threshold so each paragraph should be its own page
        page_list, line_to_page = txt_to_page_list(content, chars_per_page=60)

        # Should have 3 pages, one per paragraph
        assert len(page_list) == 3
        assert "A" * 50 in page_list[0][0]
        assert "B" * 50 in page_list[1][0]
        assert "C" * 50 in page_list[2][0]

    def test_line_to_page_mapping(self):
        """Test that line-to-page mapping is correct."""
        content = "Line 1\n\nLine 2\n\nLine 3"
        page_list, line_to_page = txt_to_page_list(content, chars_per_page=10)

        assert len(line_to_page) > 0
        assert all(isinstance(p, int) and p >= 1 for p in line_to_page)

    def test_empty_content(self):
        """Test handling of empty content."""
        page_list, line_to_page = txt_to_page_list("")
        assert page_list == []
        assert line_to_page == []


class TestPageIndices:
    """Tests for page index assignment to nodes."""

    def test_add_page_indices_basic(self):
        """Test basic page index assignment."""
        tree = [
            {'title': 'A', 'text': '', 'line_num': 1, 'nodes': []},
            {'title': 'B', 'text': '', 'line_num': 10, 'nodes': []},
        ]
        line_to_page = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3]  # Lines 1-5: page 1, 6-10: page 2, 11-13: page 3

        result = add_page_indices_to_nodes(tree, line_to_page, total_pages=3)

        assert result[0]['physical_index'] == 1
        assert result[0]['start_index'] == 1
        assert result[1]['physical_index'] == 2
        assert result[1]['start_index'] == 2

    def test_nested_nodes_get_indices(self):
        """Test that nested nodes get page indices."""
        tree = [
            {
                'title': 'Book 1',
                'text': '',
                'line_num': 1,
                'nodes': [
                    {'title': 'Chapter 1', 'text': '', 'line_num': 5, 'nodes': []},
                    {'title': 'Chapter 2', 'text': '', 'line_num': 10, 'nodes': []},
                ]
            },
        ]
        line_to_page = [1] * 5 + [2] * 5 + [3] * 5  # 15 lines, 3 pages

        result = add_page_indices_to_nodes(tree, line_to_page, total_pages=3)

        # Check parent
        assert 'physical_index' in result[0]
        # Check children
        assert 'physical_index' in result[0]['nodes'][0]
        assert 'physical_index' in result[0]['nodes'][1]

    def test_end_index_computation(self):
        """Test that end_index is computed correctly."""
        tree = [
            {'title': 'A', 'text': '', 'line_num': 1, 'nodes': []},
            {'title': 'B', 'text': '', 'line_num': 6, 'nodes': []},
            {'title': 'C', 'text': '', 'line_num': 11, 'nodes': []},
        ]
        line_to_page = [1] * 5 + [2] * 5 + [3] * 5

        result = add_page_indices_to_nodes(tree, line_to_page, total_pages=3)

        # Node A should end before node B starts
        assert result[0]['end_index'] >= result[0]['start_index']
        # Last node should extend to total_pages
        assert result[2]['end_index'] == 3

    def test_out_of_range_line_number(self):
        """Test handling of out-of-range line numbers (edge case)."""
        tree = [
            {'title': 'A', 'text': '', 'line_num': 100, 'nodes': []},  # Beyond line_to_page
        ]
        line_to_page = [1] * 5

        result = add_page_indices_to_nodes(tree, line_to_page, total_pages=1)

        # Should default to page 1 when line_num is out of range
        assert result[0]['physical_index'] == 1

    def test_deeply_nested_children_get_correct_end_index(self):
        """Test that deeply nested children inherit correct end boundaries."""
        tree = [
            {
                'title': 'Book 1',
                'text': '',
                'line_num': 1,
                'nodes': [
                    {
                        'title': 'Chapter 1',
                        'text': '',
                        'line_num': 3,
                        'nodes': [
                            {'title': 'Section 1.1', 'text': '', 'line_num': 4, 'nodes': []},
                            {'title': 'Section 1.2', 'text': '', 'line_num': 6, 'nodes': []},
                        ]
                    },
                    {'title': 'Chapter 2', 'text': '', 'line_num': 8, 'nodes': []},
                ]
            },
            {'title': 'Book 2', 'text': '', 'line_num': 10, 'nodes': []},
        ]
        line_to_page = [1] * 3 + [2] * 3 + [3] * 3 + [4] * 3  # 12 lines, 4 pages

        result = add_page_indices_to_nodes(tree, line_to_page, total_pages=4)

        # Section 1.1 should end before Section 1.2
        section_1_1 = result[0]['nodes'][0]['nodes'][0]
        section_1_2 = result[0]['nodes'][0]['nodes'][1]
        assert section_1_1['end_index'] <= section_1_2['start_index']

        # Last child of Chapter 1 (Section 1.2) should end before Chapter 2
        chapter_2 = result[0]['nodes'][1]
        assert section_1_2['end_index'] <= chapter_2['start_index']

    def test_sibling_next_line_out_of_range(self):
        """Test handling when sibling's line_num exceeds line_to_page length."""
        tree = [
            {'title': 'A', 'text': '', 'line_num': 1, 'nodes': []},
            {'title': 'B', 'text': '', 'line_num': 100, 'nodes': []},  # Out of range
        ]
        line_to_page = [1] * 5

        result = add_page_indices_to_nodes(tree, line_to_page, total_pages=1)

        # Node A should still get a valid end_index
        assert result[0]['end_index'] == 1  # Falls back to total_pages

    def test_child_inherits_parent_next_line(self):
        """Test that child nodes inherit boundary from parent's next sibling."""
        tree = [
            {
                'title': 'Book 1',
                'text': '',
                'line_num': 1,
                'nodes': [
                    {'title': 'Chapter 1', 'text': '', 'line_num': 2, 'nodes': []},
                ]
            },
            {'title': 'Book 2', 'text': '', 'line_num': 6, 'nodes': []},
        ]
        line_to_page = [1] * 3 + [2] * 3

        result = add_page_indices_to_nodes(tree, line_to_page, total_pages=2)

        # Chapter 1 should end before Book 2 starts
        chapter_1 = result[0]['nodes'][0]
        book_2 = result[1]
        assert chapter_1['end_index'] <= book_2['start_index']

    def test_last_child_with_no_parent_next_line(self):
        """Test handling of last child when parent has no next sibling."""
        tree = [
            {
                'title': 'Only Book',
                'text': '',
                'line_num': 1,
                'nodes': [
                    {'title': 'Only Chapter', 'text': '', 'line_num': 2, 'nodes': []},
                ]
            },
        ]
        line_to_page = [1] * 5

        result = add_page_indices_to_nodes(tree, line_to_page, total_pages=1)

        # Last child in last parent should extend to total_pages
        only_chapter = result[0]['nodes'][0]
        assert only_chapter['end_index'] == 1


class TestTxtToTreeAsync:
    """Tests for the async txt_to_tree function - critical for RAG integration."""

    @pytest.fixture
    def sample_book_file(self, tmp_path):
        """Create a sample book file for testing."""
        content = """FIRST BOOK

THE LIFE OF THE KING

This is the introduction to the first book.
It describes the early life of the king.

CHAPTER I

The birth of the king occurred in a small village.
The people rejoiced at his arrival.

CHAPTER II

The king grew wise and just.
His reign brought peace to the land.

SECOND BOOK

THE CRUSADE

This book describes the holy crusade.

CHAPTER I

The journey began at dawn.
Many knights joined the cause."""

        book_file = tmp_path / "sample_book.txt"
        book_file.write_text(content, encoding='utf-8')
        return str(book_file)

    @pytest.mark.asyncio
    async def test_txt_to_tree_basic(self, sample_book_file):
        """Test basic tree generation from txt file."""
        from page_index_txt import txt_to_tree

        result = await txt_to_tree(sample_book_file)

        assert 'doc_name' in result
        assert 'structure' in result
        assert result['doc_name'] == 'sample_book'
        assert len(result['structure']) >= 2  # At least 2 books

    @pytest.mark.asyncio
    async def test_txt_to_tree_with_node_text(self, sample_book_file):
        """Test that if_add_node_text='yes' includes page_list."""
        from page_index_txt import txt_to_tree

        result = await txt_to_tree(
            sample_book_file,
            if_add_node_text='yes'
        )

        # Should have page_list for RAG compatibility
        assert 'page_list' in result
        assert isinstance(result['page_list'], list)
        assert len(result['page_list']) >= 1

        # Each page should be (text, token_count) tuple
        for page in result['page_list']:
            assert isinstance(page, tuple)
            assert len(page) == 2
            assert isinstance(page[0], str)
            assert isinstance(page[1], int)

    @pytest.mark.asyncio
    async def test_txt_to_tree_nodes_have_page_indices(self, sample_book_file):
        """Test that nodes have page indices when if_add_node_text='yes'."""
        from page_index_txt import txt_to_tree

        result = await txt_to_tree(
            sample_book_file,
            if_add_node_text='yes'
        )

        def check_node_indices(node):
            assert 'physical_index' in node, f"Node {node.get('title')} missing physical_index"
            assert 'start_index' in node, f"Node {node.get('title')} missing start_index"
            assert 'end_index' in node, f"Node {node.get('title')} missing end_index"
            for child in node.get('nodes', []):
                check_node_indices(child)

        for node in result['structure']:
            check_node_indices(node)

    @pytest.mark.asyncio
    async def test_txt_to_tree_with_node_id(self, sample_book_file):
        """Test that nodes have IDs when if_add_node_id='yes'."""
        from page_index_txt import txt_to_tree

        result = await txt_to_tree(
            sample_book_file,
            if_add_node_id='yes'
        )

        def check_node_id(node):
            assert 'node_id' in node
            for child in node.get('nodes', []):
                check_node_id(child)

        for node in result['structure']:
            check_node_id(node)

    @pytest.mark.asyncio
    async def test_txt_to_tree_chars_per_page(self, sample_book_file):
        """Test that chars_per_page affects page count."""
        from page_index_txt import txt_to_tree

        # Small page size = more pages
        result_small = await txt_to_tree(
            sample_book_file,
            if_add_node_text='yes',
            chars_per_page=100
        )

        # Large page size = fewer pages
        result_large = await txt_to_tree(
            sample_book_file,
            if_add_node_text='yes',
            chars_per_page=10000
        )

        assert len(result_small['page_list']) >= len(result_large['page_list'])


class TestExtractNodeTextContent:
    """Tests for extract_node_text_content function."""

    def test_extracts_text_between_headings(self):
        """Test that text is extracted between consecutive headings."""

        node_list = [
            {'node_title': 'Chapter 1', 'line_num': 1, 'level': 1, 'type': 'chapter'},
            {'node_title': 'Chapter 2', 'line_num': 5, 'level': 1, 'type': 'chapter'},
        ]
        txt_lines = [
            'Chapter 1',
            'Content line 1',
            'Content line 2',
            '',
            'Chapter 2',
            'More content',
        ]

        result = extract_node_text_content(node_list, txt_lines)

        assert len(result) == 2
        assert 'Content line 1' in result[0]['text']
        assert 'Content line 2' in result[0]['text']
        assert 'More content' in result[1]['text']

    def test_filters_ocr_artifacts_from_content(self):
        """Test that OCR artifacts are removed from node content."""

        node_list = [
            {'node_title': 'Chapter 1', 'line_num': 1, 'level': 1, 'type': 'chapter'},
        ]
        txt_lines = [
            'Chapter 1',
            'Real content',
            'Digitized by Google',  # Should be filtered
            '123',  # Page number, should be filtered
            'More real content',
        ]

        result = extract_node_text_content(node_list, txt_lines)

        assert 'Digitized by Google' not in result[0]['text']
        assert '123' not in result[0]['text']
        assert 'Real content' in result[0]['text']
        assert 'More real content' in result[0]['text']


class TestRAGIntegration:
    """Integration tests for RAG functionality - ensures page alignment works."""

    def test_page_alignment_for_retrieval(self):
        """Test that synthetic pages align properly with content for retrieval."""
        content = """CHAPTER ONE

The king was born in 1214 in the town of Poissy.
His mother was Blanche of Castile.
She raised him with great devotion to God.

CHAPTER TWO

The king married Marguerite of Provence.
They had eleven children together.
The marriage was happy and blessed."""

        page_list, line_to_page = txt_to_page_list(content, chars_per_page=200)

        # Verify we can find content on the right pages
        assert len(page_list) >= 1

        # First chapter content should be findable
        found_king_born = False
        for page_text, _ in page_list:
            if 'born in 1214' in page_text:
                found_king_born = True
                break
        assert found_king_born, "Content 'born in 1214' not found in any page"

    def test_line_to_page_mapping_accuracy(self):
        """Test that line-to-page mapping allows finding content by line number."""
        content = "Line 1\n\nLine 2\n\nLine 3\n\nLine 4\n\nLine 5"

        page_list, line_to_page = txt_to_page_list(content, chars_per_page=20)

        # Every line should map to a valid page
        for page_num in line_to_page:
            assert 1 <= page_num <= len(page_list)


class TestDeferredSummaryMode:
    """Tests for deferred summary mode (if_add_node_summary='deferred')."""

    @pytest.fixture
    def sample_book_file(self, tmp_path):
        """Create a sample book file for testing."""
        content = """FIRST BOOK

THE LIFE OF THE KING

This is the introduction to the first book.
It describes the early life of the king.

CHAPTER I

The birth of the king occurred in a small village.
The people rejoiced at his arrival.

CHAPTER II

The king grew wise and just.
His reign brought peace to the land."""

        book_file = tmp_path / "sample_book.txt"
        book_file.write_text(content, encoding='utf-8')
        return str(book_file)

    def test_mark_nodes_for_deferred_summary_sets_fields(self):
        """mark_nodes_for_deferred_summary adds needs_summary, summary, summary_prompt."""
        from page_index_txt import mark_nodes_for_deferred_summary

        tree = [
            {'title': 'Chapter 1', 'text': 'Some content here.', 'nodes': []},
        ]

        result = mark_nodes_for_deferred_summary(tree)

        assert result[0]['needs_summary'] is True
        assert result[0]['summary'] is None
        assert 'summary_prompt' in result[0]

    def test_mark_nodes_summary_prompt_contains_title(self):
        """Summary prompt includes the node title."""
        from page_index_txt import mark_nodes_for_deferred_summary

        tree = [
            {'title': 'The Great Battle', 'text': 'Content here.', 'nodes': []},
        ]

        result = mark_nodes_for_deferred_summary(tree)

        assert 'The Great Battle' in result[0]['summary_prompt']

    def test_mark_nodes_summary_prompt_contains_text_preview(self):
        """Summary prompt includes text preview."""
        from page_index_txt import mark_nodes_for_deferred_summary

        tree = [
            {'title': 'Chapter', 'text': 'The king went to war.', 'nodes': []},
        ]

        result = mark_nodes_for_deferred_summary(tree)

        assert 'The king went to war.' in result[0]['summary_prompt']

    def test_mark_nodes_truncates_long_text(self):
        """Long text is truncated to 2000 chars with ellipsis."""
        from page_index_txt import mark_nodes_for_deferred_summary

        long_text = 'x' * 3000
        tree = [
            {'title': 'Chapter', 'text': long_text, 'nodes': []},
        ]

        result = mark_nodes_for_deferred_summary(tree)

        assert len(result[0]['summary_prompt']) < len(long_text) + 200
        assert '...' in result[0]['summary_prompt']

    def test_mark_nodes_skips_empty_text(self):
        """Nodes without text don't get summary fields."""
        from page_index_txt import mark_nodes_for_deferred_summary

        tree = [
            {'title': 'Empty', 'text': '', 'nodes': []},
            {'title': 'No text field', 'nodes': []},
        ]

        result = mark_nodes_for_deferred_summary(tree)

        assert 'needs_summary' not in result[0]
        assert 'needs_summary' not in result[1]

    def test_mark_nodes_processes_nested_children(self):
        """Nested child nodes are also marked."""
        from page_index_txt import mark_nodes_for_deferred_summary

        tree = [
            {
                'title': 'Book 1',
                'text': 'Book content',
                'nodes': [
                    {'title': 'Chapter 1', 'text': 'Chapter content', 'nodes': []},
                ],
            },
        ]

        result = mark_nodes_for_deferred_summary(tree)

        assert result[0]['needs_summary'] is True
        assert result[0]['nodes'][0]['needs_summary'] is True

    @pytest.mark.asyncio
    async def test_txt_to_tree_deferred_mode(self, sample_book_file):
        """txt_to_tree with if_add_node_summary='deferred' marks nodes."""
        from page_index_txt import txt_to_tree

        result = await txt_to_tree(
            sample_book_file,
            if_add_node_summary='deferred',
            if_add_node_text='yes',
        )

        # Check that nodes have deferred summary fields
        def check_deferred_fields(node):
            if node.get('text'):
                assert 'needs_summary' in node
                assert node['needs_summary'] is True
                assert node['summary'] is None
                assert 'summary_prompt' in node
            for child in node.get('nodes', []):
                check_deferred_fields(child)

        for node in result['structure']:
            check_deferred_fields(node)

    @pytest.mark.asyncio
    async def test_txt_to_tree_deferred_no_llm_calls(self, sample_book_file):
        """Deferred mode doesn't make any LLM calls."""
        from page_index_txt import txt_to_tree

        # This should complete without any external dependencies
        # If it tried to call LLM, it would fail since we haven't
        # configured any provider
        result = await txt_to_tree(
            sample_book_file,
            if_add_node_summary='deferred',
            if_add_node_text='yes',
        )

        # Just verify it returned something
        assert 'structure' in result
        assert len(result['structure']) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
