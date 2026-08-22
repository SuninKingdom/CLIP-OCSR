# Benchmark Representation Conversion

These utilities normalize outputs from comparison OCSR systems into the
pseudo-SMILES representation expected by the CLIP-OCSR Markush graphical
evaluator. Conversion and evaluation are deliberately separate: these scripts
do not calculate accuracy and do not remove failed samples from a benchmark.

| Utility | Primary use in this work |
|---------|--------------------------|
| `cxsmiles_to_pseudo_smiles.py` | Conservative conversion of one CXSMILES string, primarily for MarkushGrapher-2 output |
| `convert_cxsmiles_table.py` | Ordered batch conversion of MarkushGrapher-2 JSONL/CSV output or CXSMILES label tables |
| `molfile_to_pseudo_smiles.py` | Experiment-compatible conversion of V2000 MOL predictions from MolScribe, MolNexTR, and the standalone ChemEAGLE Image2Graph component |

All tools use only the Python standard library and RDKit, which is already a
CLIP-OCSR dependency.

## CXSMILES conversion

Convert and inspect one value:

```bash
python scripts/benchmark_conversion/cxsmiles_to_pseudo_smiles.py \
  '*C(*)=O |$R1;;R2;$|' --json
```

The converter reports one of four statuses:

- `success`: the represented CXSMILES features have a deterministic mapping;
- `review_required`: a candidate is returned, but some CXSMILES semantics are
  not retained by the pseudo-SMILES dialect;
- `unsupported`: an exact or defensible candidate cannot be generated; and
- `invalid`: the source cannot be parsed safely.

For example, CXSMILES position-variation target sets contain more information
than a `$` pseudo-SMILES marker. Such a projection is therefore marked
`review_required`, not silently promoted to exact ground truth.

### MarkushGrapher-2 prediction tables

MarkushGrapher-2 inference commonly produces JSONL records with `id` and
`cxsmiles` fields. Convert them while retaining every input record and its
diagnostics as follows:

```bash
python scripts/benchmark_conversion/convert_cxsmiles_table.py \
  /path/to/predictions_1000.jsonl \
  evaluation/markushgrapher2_pseudo_smiles.csv \
  --cxsmiles-column cxsmiles \
  --image-column id \
  --output-image-column Image_Name \
  --image-suffix .png \
  --pseudo-column Predicted_SMILES
```

The input order and original fields are preserved. The output additionally
contains the pseudo-SMILES candidate, `conversion_status`,
`eligible_for_automatic_evaluation`, issue codes, full issue objects, and
conversion metadata. Missing or failed predictions remain as rows with an
empty pseudo-SMILES. By default, a `review_required` candidate is retained for
manual inspection; pass `--exact-only` to leave such candidates blank.

### USPTO-M and M2S label preparation

The same core converter was used to produce the initial automatic candidates
for the USPTO-M and M2S CXSMILES label-conversion audits:

```bash
python scripts/benchmark_conversion/convert_cxsmiles_table.py \
  /path/to/labels.json labels_automatic_conversion.csv \
  --cxsmiles-column cxsmiles \
  --image-column image_name
```

Those automatic candidates were subsequently checked manually. Accordingly,
the final reviewed columns in
[`benchmark/uspto_m_pseudo_smiles_conversion_audit.csv`](../../benchmark/uspto_m_pseudo_smiles_conversion_audit.csv)
and
[`markush_parsing/m2s_pseudo_smiles_conversion_audit.csv`](../../markush_parsing/m2s_pseudo_smiles_conversion_audit.csv)
are the evaluation labels; rerunning the automatic converter does not replace
the recorded manual decisions.

## MOL-file conversion

The MOL converter is model-independent but was organized specifically for
benchmarking MOL files predicted by
[MolScribe](https://github.com/thomas0809/MolScribe),
[MolNexTR](https://github.com/CYF2000127/MolNexTR), and the standalone
[ChemEAGLE](https://github.com/CYF2000127/ChemEagle) Image2Graph component.
It provides two explicitly separated modes:

| Mode | Intended use |
|------|--------------|
| `experiment` (default) | Reproduces the `trans2smilesR_molscribe.py` conversion algorithm used for the comparison results reported in this work |
| `strict` | Optional defensive conversion for new exploratory analyses; it was not used to obtain the reported comparison values |

In `experiment` mode, the converter first applies RDKit's original read step,
temporarily replaces MOL aliases and other non-element atom labels with the
same placeholder R atoms used in the experiments, serializes the graph, and
then restores the pseudo-atom labels. MolScribe-style alias values such as `*`
and `1*` are restored as `R` and `R1`, respectively.

Canonical SMILES serialization can vary across RDKit releases. Exact
reproduction therefore uses RDKit `2022.09.1` (PyPI package version
`rdkit==2022.9.1`), which is recorded in the output metadata. A minimal pinned
requirement is provided in
[`requirements-experiment.txt`](requirements-experiment.txt). The command-line
interface refuses to run `experiment` mode with another RDKit version unless
`--allow-rdkit-version-mismatch` is supplied explicitly; results from such an
override must not be presented as an exact reproduction.

Inspect one MOL file:

```bash
python scripts/benchmark_conversion/molfile_to_pseudo_smiles.py \
  /path/to/prediction.mol --json
```

Convert a directory and create an evaluator-ready audit CSV:

```bash
python scripts/benchmark_conversion/molfile_to_pseudo_smiles.py \
  /path/to/predicted_mols \
  --output evaluation/predicted_pseudo_smiles.csv
```

Convert a model manifest whose `mol_file` column contains absolute or
manifest-relative paths:

```bash
python scripts/benchmark_conversion/molfile_to_pseudo_smiles.py \
  /path/to/predictions.csv \
  --output evaluation/predicted_pseudo_smiles.csv \
  --mol-column mol_file \
  --source-image-column image_name
```

The optional strict projection can be selected explicitly:

```bash
python scripts/benchmark_conversion/molfile_to_pseudo_smiles.py \
  /path/to/predicted_mols \
  --output evaluation/predicted_pseudo_smiles_strict.csv \
  --mode strict
```

Strict mode supports V2000 atom-alias (`A`) records, `M  RGP` R-group records,
and common pseudo/query atom carriers. It sanitizes the resulting graph and
writes canonical isomeric pseudo-SMILES while preserving specified
tetrahedral and double-bond stereochemistry. V3000 input and unlabeled
wildcard atoms are reported as unsupported rather than guessed.

Both batch modes retain unsupported and invalid MOL inputs with empty
`Predicted_SMILES` values and explicit diagnostic columns. Manifest mode also
retains expected samples whose MOL prediction is missing. Directory mode can
only enumerate files that exist, so a manifest is recommended for a
self-contained conversion audit. Independently, the graphical evaluator uses
all label rows as its denominator, so an absent or failed prediction remains
incorrect rather than disappearing from the benchmark.

The historical one-off batch wrappers omitted MOL files rejected by the first
RDKit read and serialized an unresolved conversion as the literal text
`None`. The public interface changes only this bookkeeping: it retains those
cases as explicit `invalid` or `unsupported` rows with an empty prediction.
The experiment conversion algorithm for valid predictions is unchanged, and
the evaluation denominator is preserved.

## Evaluation after conversion

An experiment-mode MOL table can be passed directly to the graphical
evaluator. A CXSMILES table should first follow its recorded review policy:
use only deterministic `success` rows (for example, generate it with
`--exact-only`) or manually adjudicate the rows marked `review_required`.

```bash
python -m clip_ocsr.evaluation.markush_metrics \
  --labels /path/to/reviewed_labels.csv \
  --predictions evaluation/predicted_pseudo_smiles.csv \
  --output evaluation/metrics.json \
  --details evaluation/per_sample.jsonl
```

See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for converter
provenance and compatible upstream license notices.
