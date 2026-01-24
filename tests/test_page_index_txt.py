"""Tests for PageIndex txt file support."""

import sys
from pathlib import Path

import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'pageindex'))

from page_index_txt import (
    build_tree_from_nodes,
    detect_book_heading,
    detect_chapter_heading,
    detect_heading,
    detect_markdown_header,
    detect_ordinal_heading,
    detect_prologue_heading,
    detect_roman_numeral_heading,
    detect_separator,
    extract_nodes_from_txt,
    is_all_caps_heading,
    is_ocr_artifact,
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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
