# Data-source provenance for the citation revision

Review date: 2026-09-25. This note records a read-only audit of existing data
metadata, file headers, download receipts, and hashes. It does not add an
experiment, change a panel, or establish family isolation. No model scores were
needed for these provenance conclusions.

Paths below are relative to the experimental repository or its data archive,
unless explicitly identified as manuscript paths. They identify the checked
sources; they do not imply that every large source file is bundled with the
manuscript. Hostnames, user names, and machine-specific absolute paths are omitted.

## PDB/mmCIF collection

The experimental structures come from the existing upstream data archive at
`Dataset/raw/pdb_mmcif/`. The candidate catalog is
`catalog_v1/monomer_candidates.jsonl.gz`.

The catalog SHA256 was checked directly and agrees with the recorded value in
`reports/length_filter_calibration_20260920/execution_lock.json`, line 10:

```text
monomer_candidates.jsonl.gz
1f3b4b185dac64064ad489f0b1df1be73d21e38693f5a7a6bd66b6da40d4ba44
```

The original database download date or a dated PDB archive release was **not
found in the checked records**. The catalog summaries identify the selection
policy, but do not supply that date. The checked
`Dataset/logs/pdb_mmcif_download.log` contains only an interrupted-download
message; `pdb_mmcif_missing_retry.log` and `pdb_mmcif_gzip_check.log` are empty.
An internal copy log does not establish the original database download date.
Local filesystem modification times were not used as download evidence.

The **2021-09-30** date is the entry-release eligibility cutoff for the relevant
panels, not a database snapshot date. It is recorded in
`catalog_v1/monomer_candidates_distribution.json` as `date_cutoff` and in the
panel protocols. Inherited Train24/Dev8 are not retroactively subject to the
later panel eligibility rule.

Berman et al. (2000) can therefore support the identification of the PDB data
resource. The reproducible local identifier is the catalog hash and the archived
target/CIF bindings, not an invented dated PDB release. No official PDB download
URL was recovered from these checked download records, so none is supplied here.

## CATH annotation sources

Two different archived sources must be distinguished. The CATH feasibility audit
and its later missing-annotation check should not be described as a single
uniform CATH release.

### Integrity-verified daily-release file

Official download URL, copied from the archived download receipt:

<https://download.cathdb.info/cath/releases/daily-release/newest/cath-b-newest-all.gz>

Evidence:

- `reports/family_gradient_feasibility/download_status.json`.
- `reports/family_gradient_feasibility/daily_coverage.json`.
- Actual complete archive file:
  `native_direction_extension_20260919/cath_retry_v2/daily-all.gz`.
- Its adjacent `download_status.json` records verified content length and gzip
  CRC; its SHA256 agrees with the report copy.

```text
daily-all.gz
bytes: 4224718
SHA256: ee4379c5185a5553418e9e85756cdaeb090643395ab2e7e0c18778263b9bc6bf
HTTP response Date: Sat, 19 Sep 2026 14:01:01 GMT
HTTP Last-Modified: Wed, 15 Jan 2025 04:22:11 GMT
```

The post-download coverage record is dated
`2026-09-19T14:05:06.723826+00:00`. These records support describing the file as
retrieved on **2026-09-19**, with server-reported Last-Modified **2025-01-15**.
The latter is HTTP source metadata, not a locally inferred timestamp or a
substitute for a numbered CATH release.

The complete file was read directly: its second column contains **495,579
`v4_3_0` rows and 105,749 `putative` rows**. Consequently, calling this file
“the CATH v4.4.0 daily snapshot” would be inaccurate. Putative annotations were
not automatically accepted as validated family assignments.

The earlier `native_direction_extension_20260919/cath/` directory contains
incomplete download attempts. Its initial hashes are not the hashes of the
integrity-verified daily archive above and should not be substituted for them.

### Later chain and unclassified lists

The source files are
`audits/cath_missing_reasons_20260920/chain.txt` and
`audits/cath_missing_reasons_20260920/unclassified.txt`.
Their directly inspected headers state, respectively:

```text
FILE NAME: CathChainList.v4.4.0 / CathUnclassifiedList.v4.4.0
FILE DATE: 16.12.2024
CATH VERSION: v4.4.0
VERSION DATE: 16.12.2024
```

The hashes were checked directly and match
`reports/family_gradient_feasibility/missing_reason_audit.json`:

```text
chain.txt
bytes: 44385324
SHA256: d1db66390cbc1625264d875ab7bdbd32553418528360454ad99f2abd0e62fb93

unclassified.txt
bytes: 10307584
SHA256: ebf77aa25c76c3ae2bed3114437ea73f8cae53de62b374547947ebc8022df2f3
```

`reports/family_gradient_feasibility/README.md` explicitly distinguishes header
date **2024-12-16**, HTTP modification date **2025-01-13**, and retrieval date
**2026-09-20**. The first two are also recorded in
`missing_reason_audit.json`; the retrieval date is supported by the archived
README's bounded-audit account. The checked records do not include the original
download URLs for these two lists, so no release-specific URLs are inferred here.

Orengo et al. (1997) identifies the CATH resource; it does not itself document
these later file versions. The audit still lacks complete annotation coverage.
This provenance clarification does not establish strict superfamily isolation,
complete historical exposure accounting, or separation from foundation-model
pretraining data.

## NCBI BLAST+

The actual archived version output in
`reports/length_filter_calibration_20260920/execution_lock.json`, line 75, is:

```text
blastp: 2.17.0+
 Package: blast 2.17.0, build Jul  1 2025 08:59:18
```

Thus the precise software description is **NCBI BLAST+ 2.17.0
(blastp 2.17.0+)**. The build timestamp is not the installation or download date.

Official package URL, copied from
`reports/length_filter_calibration_20260920/tool_download.json`, line 2:

<https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/2.17.0/ncbi-blast-2.17.0+-x64-linux.tar.gz>

The same receipt records:

```text
package bytes: 296006458
package SHA256: 3888112d8207831aa47371d93583c601f058f88b5db22dc782438b039a3a411b
package MD5: bdec166721de3b55f90a3badc83538e8
archived official MD5: bdec166721de3b55f90a3badc83538e8
```

The execution lock, lines 19–20, pins the binaries:

```text
tools/blast_2_17/ncbi-blast-2.17.0+/bin/blastp
SHA256: 5ce267c04e4988c265357bfbedc64e809545b6fcfae7ff6775266fabbee8ba0e

tools/blast_2_17/ncbi-blast-2.17.0+/bin/makeblastdb
SHA256: c1ffdcf6f15d1d8d75377cc9f37afa42bc9fa06678903bad77c2641e60778fce
```

The locked search parameters and later protocol reuse are documented in
`reports/openfold_fresh96_20260923/selection_v2.md` and
`reports/protenix_fresh192_20260925/protocol.md`. Existing manuscript wording
already identifies version 2.17.0 and the main screening parameters; retaining
the actual `2.17.0+` executable version removes a minor provenance ambiguity.
It does not change the stated limitations of operational sequence screening.
