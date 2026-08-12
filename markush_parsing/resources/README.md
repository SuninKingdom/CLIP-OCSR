# Bundled fragment library

`markush_fragment_library.json` is a description/name-to-fragment mapping
migrated from an earlier internal implementation. It is bundled here
so concrete-product generation has no dependency on another repository.

- Records: 7,627
- Size: 1,064,729 bytes
- SHA-256: `9c0e53d796799aed181a0b854ab5dfae2154ffd016dc56e1a6e93f23b75c417f`
- Record fields used by this project: `Description`, `Name`, and `SMILE`
- Attachment marker in the source data: `[R]`

The resolver validates every selected fragment with RDKit, requires exactly
one terminal attachment marker, and canonicalizes/deduplicates candidates
before use. Library-backed enumeration is reported as `partial`; it is not
claimed to exhaust an open-ended chemical class.
