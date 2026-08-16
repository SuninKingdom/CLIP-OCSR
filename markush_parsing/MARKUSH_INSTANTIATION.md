# Concrete Markush instantiation

This optional post-processing stage combines the two recognition outputs

1. a backbone pseudo-SMILES, and
2. a mapping from variable labels to allowed values

into concrete, RDKit-validated molecular SMILES. It does not change OCSR, LLM
extraction, or evaluation results unless explicitly enabled. Ground-truth
pseudo-SMILES and `pseudo_smiles_all` are never used during instantiation.

## What the default workflow does

For finitely resolved inputs, the default behavior is **exhaustive**:

1. expand supported frequency variables;
2. resolve each `[R$]` position variable within its directly attached host
   ring and deduplicate symmetry-equivalent backbones;
3. resolve every structural variable to concrete candidate fragments;
4. calculate the theoretical Cartesian-product count before fragment
   assignment;
5. attempt every planned assignment unless the user explicitly supplied a
   limit;
6. validate each molecule with RDKit, convert it to canonical isomeric SMILES,
   and deduplicate the final products; and
7. write every unique product to a plain-text SMILES file.

No default product-count limit is applied. Options such as `--max-products`
are opt-in deployment safeguards. If any user-supplied limit is reached, the
result is explicitly marked `truncated: true`; truncation is never silent.

## Primary and audit outputs

In the CLI workflows, the primary scientific result is always the unique
SMILES text file:

```text
000001_example_smiles.txt
```

It has no header and contains one canonical isomeric SMILES per line:

```text
Brc1ccccc1
Clc1ccccc1
Fc1ccccc1
Ic1ccccc1
```

Three output layers are kept separate so large enumerations do not repeat
millions of full metadata records inside one JSON object.

| Output | Purpose | Default behavior |
| --- | --- | --- |
| `*_smiles.txt` | Complete unique concrete-SMILES set | Always written after a valid enumeration plan is built |
| `*_audit.jsonl` | One detailed provenance record per unique product | Controlled by `--audit-mode` and the theoretical count |
| Result/manifest and `*_summary.json` | Small sample- and run-level counts, settings, paths, hashes, warnings and status | Always retained by the CLI workflow |

When products are streamed to files, the manifest deliberately contains
`"products": []` and `"products_embedded": false`. The product count and
the exact text-file path are stored under `instantiation.output`; the actual
SMILES are in `*_smiles.txt`.

The detailed JSONL audit stores, for each retained unique product:

- canonical product SMILES;
- backbone and position-variant indices;
- inferred position assignment;
- frequency assignment; and
- the concrete candidate selected for each variable.

If different assignments produce the same canonical molecule, the text file
contains that molecule once, the detailed audit retains the first deterministic
provenance record, and `duplicate_products_removed` counts the additional
paths.

## How automatic audit selection works

Detailed audit metadata can be much larger than the SMILES-only output. The
program therefore calculates `theoretical_product_combinations` separately for
each input sample, before the final substituent Cartesian product, and uses it
only to decide whether that sample's detailed per-product audit should be
written.

The default per-sample threshold is 10,000 theoretical combinations:

| `--audit-mode` | Theoretical combinations | Files written |
| --- | ---: | --- |
| `auto` (default) | `<= --audit-threshold` | SMILES TXT + detailed audit JSONL + summary/manifest |
| `auto` (default) | `> --audit-threshold` | SMILES TXT + summary/manifest |
| `always` | Any number | SMILES TXT + detailed audit JSONL + summary/manifest |
| `never` | Any number | SMILES TXT + summary/manifest |

The threshold **does not limit product generation**. It controls only the
detailed JSONL audit. Change it with, for example,
`--audit-threshold 50000`.

The theoretical count is calculated after frequency/position backbone
generation and backbone deduplication, but before concrete fragment grafting:

```text
unique backbone variants
    x candidates for R1
    x candidates for R2
    x ...
```

It is an upper bound on final unique products. It can include assignments that
later fail chemical validation and assignments that collapse to the same final
canonical SMILES.

For example, consider `[R1]CC[R2]` when both R1 and R2 are `halogen`:

```text
backbone variants                 = 1
candidate fragments for R1        = 4
candidate fragments for R2        = 4
theoretical combinations          = 1 x 4 x 4 = 16
attempted combinations            = 16
final unique canonical products   = 10
duplicate products removed        = 6
```

The six duplicates arise because some R1/R2 assignments become the same
canonical molecule after exchanging the two terminal sites. If the audit
threshold is 10, `auto` suppresses the detailed JSONL because `16 > 10`, but
all 16 assignments are still attempted and all 10 unique SMILES are written.

## Optional enumeration limits

With no `--max-*` arguments, finitely resolved candidates are exhaustively
enumerated. A user may explicitly bound a deployment or exploratory run:

```bash
--max-products 1000000
```

`--max-products N` means at most `N` **unique canonical SMILES**, not `N` raw
assignment attempts. Reaching it sets `truncated: true`,
`enumeration_complete: false`, and normally `status: partial`.

Additional expert safeguards are also opt-in:

- `--max-assignment-attempts`
- `--max-position-variants`
- `--max-frequency-variants`
- `--max-repeat-count`
- `--max-candidates-per-value`

Every active limit and whether it was reached are recorded in the manifest and
summary. These options should not be supplied when an exhaustive paper result
is required.

## Two independent kinds of completeness

Chemical interpretation quality and enumeration completion are reported
separately.

- `enumeration_complete`: every planned combination of the resolved finite
  candidates was attempted and no limit truncated the run.
- `is_fully_enumerated`: compatibility alias for `enumeration_complete` in
  schema 2.0.
- `status: complete`: enumeration is complete and every supplied definition
  was handled by an exact mapping or a reviewed complete finite class.
- `status: partial`: valid products exist, but an inferred position,
  representative/library-backed mapping, unresolved value, invalid
  alternative, unmatched variable, parse failure, or truncation occurred.
- `status: failed`: no valid concrete product could be produced.

For example, a sample containing `[R1$]` can have
`enumeration_complete: true` because every inferred host-ring position was
processed, while still having `status: partial` because the host-ring position
set is an inference rather than an explicit patent locant constraint.

`complete` never claims that an open-ended patent class has been exhaustively
covered. It applies only to the parsed, supported and finitely resolved input
definitions.

## Two-stage deduplication

Deduplication occurs at two chemically distinct stages.

### 1. Position-variant backbone deduplication

For `[R1$]`, the placeholder is moved across chemically valid atoms of the
directly attached host ring. Canonical mapped pseudo-structures are used to
collapse symmetry-equivalent locations before any concrete R1 value is
assigned. The audit reports:

- `candidate_sites`;
- `chemically_valid_sites`;
- `valid_unique_sites`; and
- `duplicate_position_variants_removed`.

For example, a methyl-substituted benzene ring has six ring atoms, five
chemically available positions for another substituent, and three unique
ortho/meta/para products after symmetry deduplication.

### 2. Final concrete-product deduplication

All valid products are converted to canonical isomeric SMILES. Assignment
swaps that produce the same molecule are retained once in `*_smiles.txt` and
counted in `duplicate_products_removed`.

## Position variation and the `$` marker

`[R1$]` is handled as a two-stage approximation:

```text
position-variable backbone
    -> fixed-position, deduplicated backbone variants
    -> concrete R1 fragment assignment
```

The variable is enumerated only within the directly attached host ring; it is
not moved automatically to another fused, bridged, or spiro ring. If the
attachment atom belongs to more than one perceived ring and the pseudo-SMILES
does not identify which is intended, the observed position is retained and the
ambiguity is reported instead of guessing.

Patent locants such as "R1 may be located at position 5 or 6" are not stable
SMILES or RDKit atom indices. The current pseudo-SMILES and variable-table
schema therefore cannot enforce those locants without a separate
patent-locant-to-graph-atom mapping. Results involving inferred `$` positions
are consequently marked `partial` even when their inferred space was fully
enumerated.

## Supported variable behavior

| Input feature | Instantiation behavior |
| --- | --- |
| Different variable labels | Cartesian product across labels |
| Repeated occurrence of one label | One selected value is applied consistently at every occurrence |
| Explicit groups | Graph grafting through one `[*:1]` attachment atom |
| H and D | Hydrogen substitution/removal and isotopic deuterium attachment |
| Linker variables such as O, S, N or CH2 | Internal atom replacement when molecular valence permits it |
| `single bond` | Degree-two placeholder contraction |
| `[(CH2)n]`, `[O(CH2)n]` | Linear frequency expansion from integer values or ranges |
| `S...[(O)m]` | Sulfide/sulfoxide/sulfone-style `m = 0/1/2` expansion |
| `[R1$]` | Host-ring position enumeration followed by symmetry deduplication |
| Common OCSR composites such as `[OR2]` | Conservatively normalized to `O[R2]` only when `R2` is an extracted variable |
| Label typography such as `R_c`, `R^c`, `R_{c}` | Aligned to `[Rc]` only when the backbone match is unique |

Final products contain no dummy atoms. Every emitted product passes RDKit
sanitization and is written as canonical isomeric SMILES.

## Fragment resolution and project independence

Resolution uses this deterministic precedence:

1. reviewed exact mappings and complete finite classes;
2. exact normalized `Description` or `Name` lookup in the bundled
   `resources/markush_fragment_library.json`; and
3. a conservative straight-chain representative rule for simple ranged alkyl
   or alkoxy descriptions.

For example, `halogen` is completely expanded to F, Cl, Br and I. An
unrestricted class such as `aryl`, `alkyl` or `heteroaryl` is not a finite
chemical space. A representative or library-backed interpretation is therefore
reported as `partial`, never as complete coverage. Definitions that cannot be
converted without guessing remain unresolved.

The fragment library is bundled inside this repository. The default workflow
has no runtime dependency on another project or a private absolute path. Its
path, size and SHA-256 fingerprint are recorded with offline runs. Use
`--fragment-library` for an alternate file or `--no-fragment-library` to use
only the reviewed built-in rules.

## Use during a new image-pipeline run

The recognition pipeline remains unchanged unless `--instantiate` is supplied:

```bash
python markush_parsing/run.py \
  --input /path/to/images \
  --mineru-dir /path/to/mineru-output \
  --output /path/to/results \
  --llm deepseek \
  --instantiate
```

For each processed image, the sample directory contains:

```text
results/<provider>/<image-stem>/
  result.json
  concrete_smiles.txt
  concrete_smiles_audit.jsonl   # auto: only when theoretical count <= 10,000
```

`result.json` contains the lightweight instantiation summary and file hashes.
The global `per_sample.jsonl` contains the same summary, not the full product
list.

Examples:

```bash
# Force detailed audit even for a large enumeration
python markush_parsing/run.py ... --instantiate --audit-mode always

# Keep only SMILES plus the lightweight result/summary
python markush_parsing/run.py ... --instantiate --audit-mode never

# Explicitly request at most one million unique products
python markush_parsing/run.py ... --instantiate --max-products 1000000
```

Checkpoint resume does not rewrite completed samples. Use the offline command
below for an existing checkpoint or start a separate `--no-resume` run.

## Instantiate an existing saved experiment

No image processing, model inference or API request is required:

```bash
python markush_parsing/run.py \
  --instantiate-results /path/to/per_sample.jsonl \
  --instantiation-output /path/to/instantiated_manifest.jsonl \
  --instantiation-products-dir /path/to/instantiated_products
```

The resulting layout is:

```text
instantiated_manifest.jsonl
instantiated_manifest_summary.json
instantiated_products/
  000001_sample_name_smiles.txt
  000001_sample_name_audit.jsonl   # depends on audit mode/threshold
  000002_sample_name_smiles.txt
  ...
```

The source JSONL is never overwritten. Existing outputs are protected unless
`--overwrite-instantiation-output` is explicitly supplied. The manifest
accepts ordinary pipeline fields (`predicted_smiles`, `predicted_variables`)
and variable-only rerun fields (`source_predicted_smiles`,
`source_predicted_variables`).

The run summary records:

- source and manifest SHA-256 hashes;
- product directory and per-sample product paths/hashes;
- which source fields were used;
- theoretical, attempted, invalid, duplicate and final-unique counts;
- complete/partial/failed, enumeration-complete and truncated counts;
- audit mode, threshold and number of detailed audits written;
- every active user limit;
- fragment-library metadata;
- Python, RDKit and output-schema versions; and
- the exact command arguments.

## Python API

Small in-memory use remains available for compatibility:

```python
from markush_parsing.fragment_resolver import FragmentResolver
from markush_parsing.markush_instantiator import MarkushInstantiator

result = MarkushInstantiator(FragmentResolver()).instantiate(
    "c1ccccc1[R1]",
    {"R1": ["halogen"]},
)
smiles = [item["smiles"] for item in result["products"]]
```

For large enumerations, stream products to files instead of embedding full
records in memory:

```python
result = MarkushInstantiator(FragmentResolver()).instantiate(
    "c1ccccc1[R1]",
    {"R1": ["halogen"]},
    products_path="products.txt",
    audit_path="products_audit.jsonl",
    audit_mode="auto",
    audit_threshold=10_000,
)
```

The file mode streams product and optional audit lines. Exact final
deduplication retains only the set of canonical SMILES keys in memory, so
memory grows with the number of unique products rather than with full
per-product metadata. Users requesting millions of unique structures should
still plan sufficient RAM and disk space. A deliberately limited pilot run can
be used to obtain the theoretical count before launching an unrestricted run;
the limit affects enumeration, not calculation of that count.

## Known boundaries

- Open-ended patent language cannot be exhaustively instantiated without a
  formally bounded fragment ontology.
- Upstream OCSR or variable-extraction errors necessarily propagate to the
  generated chemical space.
- Patent locants require a locant-to-atom mapping not present in the current
  representation.
- Repeated variable-bearing units such as `[(R1)x]` are not expanded because
  their graph attachment semantics are not uniquely defined by the current
  LLM output schema.
- Cross-variable logical constraints such as "R1 and R2 together form a ring"
  require a richer constraint schema and remain unresolved rather than guessed.

## Regression test

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:markush_parsing \
  python -m unittest -v markush_parsing.tests.test_markush_instantiator
```
