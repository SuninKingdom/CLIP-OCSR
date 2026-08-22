# Markush Benchmark Data

This directory contains source-level metadata for the three Markush benchmark test sets used in CLIP-OCSR evaluation. Each test set consists of 101 patent-derived images and 101 journal-derived images (202 images per variation type).

It also includes the reviewed USPTO-M CXSMILES-to-pseudo-SMILES conversion
audit described below.

Representation-normalization tools used for benchmark comparisons are
documented in
[`scripts/benchmark_conversion/README.md`](../scripts/benchmark_conversion/README.md).
They convert MarkushGrapher-2 CXSMILES predictions and V2000 MOL outputs from
MolScribe, MolNexTR, or ChemEAGLE while retaining conversion failures as
explicit empty predictions.

## Test Sets

| Test Set | Description | Patent Source | Journal Source |
|----------|-------------|---------------|----------------|
| Markush-SubVar202 | Substituent variation | `MarkushWithSubstituentVariation_Patent.csv` | `MarkushWithSubstituentVariation_Journal.csv` |
| Markush-FreVar202 | Frequency variation | `MarkushWithFrequencyVariation_Patent.csv` | `MarkushWithFrequencyVariation_Journal.csv` |
| Markush-PosVar202 | Position variation | `MarkushWithPositionVariation_Patent.csv` | `MarkushWithPositionVariation_Journal.csv` |

## USPTO-M Reviewed Label Conversion

`uspto_m_pseudo_smiles_conversion_audit.csv` records the reviewed conversion of
the 74 USPTO-M CXSMILES labels into the pseudo-SMILES representation used for
Markush graphical evaluation. The original row order and conversion audit
fields are retained.

The conversion resource contains 23 structures that cannot be represented by
the current pseudo-SMILES notation. Their `final_pseudo_smiles` values are
intentionally empty, and the reasons are recorded in `manual_review_status`
and `review_notes`. These empty values are conversion limitations, not missing
OCSR predictions, and should not be treated as valid pseudo-SMILES labels.

The initial `pseudo_smiles` candidates were generated with
[`cxsmiles_to_pseudo_smiles.py`](../scripts/benchmark_conversion/cxsmiles_to_pseudo_smiles.py)
and then manually checked. The reviewed `final_pseudo_smiles` and status fields
in this audit file remain authoritative for evaluation.

### Source, attribution, and license

The original USPTO-M annotations were obtained from the `uspto-markush` test
split of the
[MarkushGrapher-2 Datasets](https://huggingface.co/datasets/docling-project/MarkushGrapher-2-Datasets),
which is released under the
[Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
This CSV is an adapted resource: the original CXSMILES annotations were
converted to pseudo-SMILES, manually reviewed, and supplemented with conversion
status and review notes. The derived CSV is distributed under CC BY 4.0.

When using or redistributing this CSV, credit the original dataset and its
authors, link to CC BY 4.0, indicate that the CXSMILES annotations were converted
and manually reviewed in this work, and cite the associated MarkushGrapher
publication identified on the source dataset page. This data-specific notice
applies to the derived CSV; the repository software remains under the MIT
License.

### Reviewed-conversion columns

| Column | Description |
|--------|-------------|
| `id` | Original zero-based row identifier |
| `image_name` | Corresponding benchmark image filename |
| `cxsmiles` | Original CXSMILES annotation |
| `pseudo_smiles` | Automatically converted pseudo-SMILES, when available |
| `conversion_status` | Automatic conversion outcome |
| `final_pseudo_smiles` | Manually reviewed pseudo-SMILES used when representable |
| `manual_review_status` | Outcome of manual review |
| `review_notes` | Audit note describing confirmation, correction, or limitations |

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
