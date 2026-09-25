# Bibliography source audit — 25 September 2026

This audit records the public sources used to revise `paper_references.bib`. Existing citation keys are retained, including keys whose year reflects an earlier preprint. Conference publication years are recorded in the `year` field. No experimental results were changed.

## Existing entries

| Citation key | Verified revision | Public source |
| --- | --- | --- |
| `candido2026language` | Record as a bioRxiv article/preprint; add DOI `10.64898/2026.06.03.729735`; correct the seventh author's given name to Bryan Z. Wu. The existing full author list was checked against the preprint. | [bioRxiv version 1](https://www.biorxiv.org/content/10.64898/2026.06.03.729735v1) |
| `seo2026atlasfold` | Retain the group author as `{Team KAIST}` so BibTeX treats it as a corporate name; add the version 2 URL, which includes the group author. | [bioRxiv version 2](https://www.biorxiv.org/content/10.64898/2026.09.04.749352v2) |
| `hu2021lora` | Use the ICLR 2022 conference entry rather than the 2021 arXiv entry; preserve the existing eight authors. | [Official ICLR/OpenReview paper](https://openreview.net/pdf?id=nZeVKeeFYf9), [conference record](https://openreview.net/forum?id=nZeVKeeFYf9) |
| `liu2023boft` | Use the ICLR 2024 conference entry rather than the 2023 arXiv entry. Author names follow the conference PDF; see the source discrepancy below. | [Official conference PDF](https://proceedings.iclr.cc/paper_files/paper/2024/file/a63ce8e6867a1bf4b4ca62e5077814d9-Paper-Conference.pdf), [OpenReview paper](https://openreview.net/pdf?id=7NzgkEdGyr) |
| `meng2024pissa` | Use the NeurIPS 2024 proceedings entry, volume 37, pages 121038–121072, with the official title, three authors, editors, publisher and DOI. | [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/db36f4d603cc9e3a2a5e10b93e6428f2-Abstract-Conference.html), [official BibTeX](https://proceedings.neurips.cc/paper_files/paper/2024/file/db36f4d603cc9e3a2a5e10b93e6428f2-Bibtex-Conference.bib) |
| `wu2024reft` | Use the NeurIPS 2024 proceedings entry, volume 37, pages 63908–63962, with the official title, seven authors, editors, publisher and DOI. | [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/75008a0fba53bf13b0bb3b7bff986e0e-Abstract-Conference.html), [official BibTeX](https://proceedings.neurips.cc/paper_files/paper/2024/file/75008a0fba53bf13b0bb3b7bff986e0e-Bibtex-Conference.bib) |
| `zhang2004tm` | Add issue 4 to volume 57, pages 702–710. | [Wiley publisher record](https://onlinelibrary.wiley.com/doi/abs/10.1002/prot.20264) |

The [BOFT proceedings HTML record](https://proceedings.iclr.cc/paper_files/paper/2024/hash/a63ce8e6867a1bf4b4ca62e5077814d9-Abstract-Conference.html) lists David Ha as its final author, whereas the linked conference PDF and OpenReview PDF list Bernhard Schölkopf. This revision follows the paper's PDF author list and does not copy that conflicting HTML field. ICLR entries have no proceedings page range added; no range is inferred from PDF length.

## Added entries

| Citation key | Verified bibliographic data | Public source |
| --- | --- | --- |
| `camacho2009blastplus` | Camacho and six coauthors; *BLAST+: architecture and applications*; BMC Bioinformatics 10, article 421 (2009); DOI `10.1186/1471-2105-10-421`. | [Publisher](https://doi.org/10.1186/1471-2105-10-421), [NCBI article record](https://pubmed.ncbi.nlm.nih.gov/20003500/) |
| `holm1979simple` | Sture Holm; *A Simple Sequentially Rejective Multiple Test Procedure*; Scandinavian Journal of Statistics 6(2), 65–70 (1979). The stable JSTOR URL is supplied rather than inferring a DOI from its identifier. | [JSTOR record](https://www.jstor.org/stable/4615733), [archived copy of the original article with its bibliographic cover](https://www.ime.usp.br/~abe/lista/pdf4R8xPVzCnX.pdf) |
| `kingma2015adam` | Diederik P. Kingma and Jimmy Ba; *Adam: A Method for Stochastic Optimization*; ICLR 2015. The author-deposited record explicitly identifies the 2015 conference publication, although the initial arXiv submission was in 2014. | [Author-deposited paper and publication note](https://arxiv.org/abs/1412.6980) |
| `berman2000pdb` | Berman and seven coauthors; *The Protein Data Bank*; Nucleic Acids Research 28(1), 235–242 (2000); DOI `10.1093/nar/28.1.235`. | [Original article in PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC102472/), [publisher DOI](https://doi.org/10.1093/nar/28.1.235) |
| `orengo1997cath` | Orengo and five coauthors; *CATH—a hierarchic classification of protein domain structures*; Structure 5(8), 1093–1108 (1997); DOI `10.1016/S0969-2126(97)00260-8`. Author given names use the initials supplied by the indexed article record. | [NCBI article record](https://pubmed.ncbi.nlm.nih.gov/9309224/), [publisher DOI](https://doi.org/10.1016/S0969-2126(97)00260-8) |
| `qiu2023oft` | Qiu and eight coauthors; *Controlling Text-to-Image Diffusion by Orthogonal Finetuning*; NeurIPS 2023, volume 36, pages 79320–79362; official title, author list, editors, publisher and DOI. | [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/faacb7a4827b4d51e201666b93ab5fa7-Abstract-Conference.html), [official BibTeX](https://proceedings.neurips.cc/paper_files/paper/2023/file/faacb7a4827b4d51e201666b93ab5fa7-Bibtex-Conference.bib) |
| `lezcano2019orthogonal` | Mario Lezcano-Casado and David Martínez-Rubio; *Cheap Orthogonal Constraints in Neural Networks: A Simple Parametrization of the Orthogonal and Unitary Group*; ICML 2019, PMLR 97, 3794–3803; editors Kamalika Chaudhuri and Ruslan Salakhutdinov. | [PMLR official paper and BibTeX](https://proceedings.mlr.press/v97/lezcano-casado19a.html) |

## Metadata conventions

The BLAST+ value 421 is an article number represented using the existing BibTeX style's `pages` field. Proper names and acronyms that must retain capitalization are protected with braces. Publisher DOI links and official conference records are preferred to secondary bibliography websites. NeurIPS page ranges, editors and publisher were checked against the downloadable official BibTeX records. No unsupported venue, page range, author expansion or publication year is supplied.

Validation: BibTeX processed all 21 unique entries successfully with the standard `plain` bibliography style; no errors or warnings were reported.
