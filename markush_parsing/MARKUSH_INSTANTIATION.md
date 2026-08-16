# Concrete Markush instantiation

This optional stage converts the two existing pipeline outputs

1. a backbone pseudo-SMILES, and
2. a mapping from variable labels to allowed values

into validated concrete molecular SMILES. It is implemented as a post-processing
stage and does not change OCSR, LLM extraction, or the published evaluation
metrics unless it is explicitly enabled.

## Design and independence

The implementation uses two established ideas:

- description/name-to-fragment lookup from the bundled
  `resources/markush_fragment_library.json`; and
- Cartesian products across independently defined variable labels.

An earlier internal combiner performed textual SMILES replacement. This implementation
operates on RDKit molecular graphs, validates every product, preserves atom,
bond, tetrahedral and alkene stereochemistry, emits canonical isomeric SMILES,
deduplicates products, and records every approximation or enumeration limit.

The fragment JSON is bundled inside this project, so the default workflow has
no runtime dependency on any other repository. Its path
and SHA-256 fingerprint are stored with every offline run. A different copy can
still be supplied with `--fragment-library` or
`MARKUSH_FRAGMENT_LIBRARY`. `--no-fragment-library` restricts resolution to
the reviewed built-in mappings and deterministic range rules.

## Supported behavior

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
| `[R1$]` | Candidate sites inferred only within the directly attached host ring; symmetry-equivalent backbones are deduplicated before substituent assignment |
| Common OCSR composites such as `[OR2]` | Conservatively normalized to `O[R2]` only when `R2` is an extracted variable |
| Label typography such as `R_c`, `R^c`, `R_{c}` | Aligned to `[Rc]` only when the backbone match is unique |

Final products contain no dummy atoms. Every product passes RDKit
sanitization and is written as canonical isomeric SMILES.

## Fragment resolution and completeness

Resolution is deterministic and uses this precedence:

1. reviewed exact mappings and finite classes;
2. exact normalized `Description` or `Name` lookup in the bundled fragment
   library; and
3. a conservative straight-chain representative rule for simple ranged alkyl
   or alkoxy descriptions.

For example, `halogen` is completely expanded to F, Cl, Br and I. In contrast,
an unrestricted class such as `aryl`, `alkyl` or `heteroaryl` is not a finite
chemical space. A representative or library-backed expansion is therefore
reported as `partial`, never as complete coverage. Descriptions that cannot be
converted without guessing remain unresolved.

The three statuses have strict meanings:

- `complete`: all supplied values were exact or a reviewed complete finite
  class, every requested variation was enumerated, and no limit was reached.
- `partial`: valid products were produced, but a representative/library-backed
  mapping, inferred position, unresolved/incompatible alternative, unmatched
  variable, partial backbone parse, or enumeration limit was involved.
- `failed`: no valid concrete product could be produced. `failure_reason`,
  `unresolved_core_labels`, `unresolved_values`, `errors`, and `warnings`
  retain the cause.

`is_fully_enumerated` is true only for `complete` results.

## Use during a new pipeline run

The default pipeline is unchanged. Add `--instantiate` to enable the stage:

```bash
python markush_parsing/run.py \
  --input /path/to/images \
  --mineru-dir /path/to/mineru-output \
  --llm deepseek \
  --instantiate \
  --max-products 256
```

Each sample receives an additional `instantiation` object in `result.json` and
`per_sample.jsonl`. Failure of this optional stage is isolated from OCSR and
variable-extraction evaluation.

Checkpoint resume does not rewrite previously completed rows. If an existing
checkpoint predates this feature, use the offline command below (preferred) or
start a separate `--no-resume` run.

## Instantiate an existing saved experiment

No image processing, model inference, or API request is needed:

```bash
python markush_parsing/run.py \
  --instantiate-results /path/to/per_sample.jsonl \
  --instantiation-output /path/to/per_sample_instantiated.jsonl \
  --max-products 256
```

The source JSONL is never overwritten. Existing output is also protected; use
`--overwrite-instantiation-output` only for an intentional replacement.

The command accepts both ordinary pipeline fields (`predicted_smiles`,
`predicted_variables`) and the variable-only rerun provenance fields
(`source_predicted_smiles`, `source_predicted_variables`). The accompanying
`*_summary.json` records:

- source/output SHA-256 hashes;
- which saved fields were used;
- complete/partial/failed counts and failure reasons;
- all enumeration limits;
- fragment-library path, size and SHA-256;
- Python, RDKit and output-schema versions; and
- the exact command arguments.

Ground-truth pseudo-SMILES and `pseudo_smiles_all` are never consulted during
instantiation.

The same operation is available as a Python API:

```python
from markush_parsing.fragment_resolver import FragmentResolver
from markush_parsing.markush_instantiator import MarkushInstantiator

resolver = FragmentResolver()  # uses resources/markush_fragment_library.json
result = MarkushInstantiator(resolver).instantiate(
    "c1ccccc1[R1]",
    {"R1": ["halogen"]},
)
products = [item["smiles"] for item in result["products"]]
```

## Default safeguards

- maximum unique products per sample: 256
- maximum assignment attempts per sample: 10,000
- maximum inferred position variants: 64
- maximum frequency variants: 64
- maximum repeat count: 100
- maximum fragment candidates per input value: 64

All limits are exposed through `run.py --help`. Reaching any limit sets
`truncated: true`, changes the status to `partial`, and retains the theoretical
and attempted combination counts.

## Known boundaries

- Open-ended patent language cannot be exhaustively instantiated without a
  formally bounded fragment ontology.
- `$` gives no explicit allowed-site set. The current host-ring rule is an
  inference and is always labelled `partial`; it deliberately does not use
  evaluation labels to choose sites. A uniquely identified host ring is
  enumerated without crossing into another fused, bridged, or spiro ring. If
  the attachment atom belongs to more than one perceived ring, the observed
  site is retained and the ambiguity is recorded instead of guessing a ring.
- Patent locants such as "R1 may be located at position 5 or 6" cannot be
  enforced from the present pseudo-SMILES and variable-table schema because
  patent locants are not stable SMILES or RDKit atom indices. Supporting such
  constraints requires an explicit locant-to-atom mapping.
- Repeated variable-bearing units such as `[(R1)x]` are not expanded because
  their graph attachment semantics are not uniquely defined by the current
  LLM output schema.
- Cross-variable logical constraints (for example, "R1 and R2 together form a
  ring" or conditional exclusions) require a richer constraint schema and are
  retained as unresolved rather than guessed.
- Concrete-product quality is bounded by the correctness of both upstream
  pseudo-SMILES and extracted variable definitions.

## Regression test

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:markush_parsing \
  python -m unittest -v markush_parsing.tests.test_markush_instantiator
```
