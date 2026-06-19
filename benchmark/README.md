# Markush Benchmark Source Metadata

This directory contains source-level metadata for the three Markush benchmark test sets used in CLIP-OCSR evaluation. Each test set consists of 101 patent-derived images and 101 journal-derived images (202 images per variation type).

## Test Sets

| Test Set | Description | Patent Source | Journal Source |
|----------|-------------|---------------|----------------|
| Markush-SubVar202 | Substituent variation | `MarkushWithSubstituentVariation_Patent.xlsx` | `MarkushWithSubstituentVariation_Journal.xlsx` |
| Markush-FreVar202 | Frequency variation | `MarkushWithFrequencyVariation_Patent.xlsx` | `MarkushWithFrequencyVariation_Journal.xlsx` |
| Markush-PosVar202 | Position variation | `MarkushWithPositionVariation_Patent.xlsx` | `MarkushWithPositionVariation_Journal.xlsx` |

## Column Definitions

### Patent-derived images (`*_Patent` files)

| Column | Description |
|--------|-------------|
| Patent Office | Patent office (e.g., WIPO, USPTO) |
| Patent/Publication Number | Patent or publication identifier |
| Publication Date | Publication or filing date |
| Page Number | Page number within the patent document |

### Journal-derived images (`*_Journal` files)

| Column | Description |
|--------|-------------|
| PMID | PubMed ID |
| DOI | Digital Object Identifier |
| Title | Article title |
| Journal | Journal name |
| Year | Publication year |
| Page Number | Page number within the article |

