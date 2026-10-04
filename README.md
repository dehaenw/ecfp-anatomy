# Anatomy of an ECFP

[marimo](https://marimo.io) notebook on bit provenance and collisions in ECFP (Morgan) fingerprints, and their effect
on models trained and interpreted on folded fingerprints. Entry for the
[molab Notebook Competition #3: Cheminformatics Challenge](https://marimo.io/pages/events/notebook-competition-3).

## Contents

- **Anatomy figure**: every atom environment at every radius, linked to its folded bit, with redundant
  environments and collisions marked. Molecule, radius, fpSize, ECFP/FCFP invariants and chirality are adjustable.
- **Bit inspector**: all environments behind a bit, highlighted on the molecule.
- **Dataset-scale collisions** on the OpenADMET / ExpansionRx training set: fraction of molecules with collided bits,
  environments per bit, and the most frequent colliding pairs.
- **Ridge vs LightGBM** on folded and unfolded count fingerprints for eight ADMET endpoints. Exact atom attribution
  (w · count for Ridge, TreeSHAP for LightGBM) and the share of each attribution borrowed from other environments
  on the same bits.
- **Significance testing**: on-demand 5 × 5 repeated CV with repeated-measures ANOVA and Tukey HSD, following
  Ash et al. [4], for the comparative claims. The rest of the notebook uses a single random split for speed and is
  exploratory, not a benchmark.
- **Bulk physchem baselines**: eight standard descriptors and ECFP0 counts against ECFP4.

## Running

Dependencies are declared in the notebook's inline script metadata (PEP 723), so either of these works:

```bash
uvx marimo edit --sandbox ecfp_anatomy.py
```

```bash
uvx marimo run --sandbox ecfp_anatomy.py
```

On [molab](https://molab.marimo.io), add the notebook from its GitHub URL using the new-notebook dropdown.

The dataset is downloaded from Hugging Face at startup. The dataset and model sections need a network connection;
the single-molecule sections do not.

## Data

[OpenADMET / ExpansionRx blind challenge training set](https://huggingface.co/datasets/openadmet/openadmet-expansionrx-challenge-train-data),
licensed CC BY 4.0.

## References

1. M. Praski, J. Adamczyk, W. Czech. *Benchmarking Pretrained Molecular Embedding Models For Molecular
   Representation Learning.* arXiv:2508.06199 (2025).
2. M. Torrisi, S. Asadollahi, A. de la Vega de León, K. Wang, W. Copeland. *Do chemical language models provide a
   better compound representation?* NeurIPS 2023 Workshop on New Frontiers of AI for Drug Discovery and
   Development (2023). https://doi.org/10.1101/2023.11.07.566025
3. J. Deng, Z. Yang, H. Wang, I. Ojima, D. Samaras, F. Wang. *A systematic study of key elements underlying
   molecular property prediction.* Nat. Commun. 14, 6395 (2023).
4. J. R. Ash, C. Wognum, R. Rodríguez-Pérez, M. Aldeghi, A. C. Cheng, D.-A. Clevert, O. Engkvist, C. Fang,
   D. J. Price, J. M. Hughes-Oliver, W. P. Walters. *Practically Significant Method Comparison Protocols for
   Machine Learning in Small Molecule Drug Discovery.* J. Chem. Inf. Model. 65, 9398–9411 (2025).

## License

Code: MIT (see [LICENSE](LICENSE)). The ExpansionRx data is not included in this repository; it is
downloaded at runtime and remains under its own CC BY 4.0 license.

## AI disclosure

The visualization was originally hand-written for a paper. The marimo conversion and the dataset and modeling
sections were written with help from Claude (Anthropic).
