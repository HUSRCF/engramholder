# Working agreement

On 2026-09-22 the user explicitly resumed writing in a separate writing branch,
approved the central claim and contribution order, and authorized replacing the
ICLR example text with section scaffolding and placeholders. A subsequent review
authorized filling the manuscript, generating figures/tables from verified JSON,
and preparing a bounded anonymous artifact. The prior writing
pause is superseded for this authorized scope.

Start with README.md, notes/workflow.md, and reports/evidence_review.md; use
reports/results_update_20260922.md for newer completed results. Active writing
plan: notes/writing_branch_20260922/structure_and_opening.md. Entry point:
iclr2027_conference.tex with sections/ and appendices/.

Keep the official style files unchanged, preserve anonymous submission mode,
and distinguish drafting placeholders from verified results. Separate directly
checked artifacts, historical summaries, and interpretations. Preserve negative
results and pending status. Do not infer a pending experiment's outcome.

Author policy (2026-09-24, latest user instruction): keep `main` anonymous.
Author names, affiliations, PI relationships and named draft sources stay local,
outside tracked files. Do not create or push an author branch. Preserve this
restriction when committing manuscript revisions and packaging source artifacts.

The writing branch does not alter experiment contracts, running jobs, or result
artifacts. Standing user authorization (2026-09-23): after each completed manuscript revision,
run the relevant checks, commit the scoped changes, and push to origin/main without
waiting for another request. Report push failures honestly; never force-push or
include unrelated changes. This supersedes the earlier per-revision push requirement. The original template is
backed up under notes/writing_branch_20260922/iclr2027_original_template.tex.
notes/unreviewed_writing_20260921 remains archival, not the active manuscript.

Current manuscript status: reports/manuscript_fill_20260922.md. Generated numbers
come from scripts/build_paper_assets.py with a locked input hash manifest; do not
manually edit numeric outputs or silently update the evidence lock.
