# Bundled fragment library

`markush_fragment_library.json` is a description/name-to-fragment mapping
migrated from an earlier internal implementation. It is bundled here
so concrete-product generation has no dependency on another repository.

- Records: 7,627
- Size: 782,722 bytes
- SHA-256: `e42a93b7cfbe17e350dab97e68bc97019a75fca18dbdf65021d8997260bcded7`
- Record fields used by this project: `Description`, `Name`, and `SMILES`
- Attachment marker in the source data: `[R]`

The resolver validates every selected fragment with RDKit, requires exactly
one terminal attachment marker, and canonicalizes/deduplicates candidates
before use. Library-backed enumeration is reported as `partial`; it is not
claimed to exhaust an open-ended chemical class.
