# Anatomy of an ECFP

An interactive [marimo](https://marimo.io) notebook that asks: when a model says "bit 1380 matters", which chemistry
is it talking about, and how often is the honest answer "several unrelated things"?

It follows a molecule through the three steps that produce an ECFP (Morgan) fingerprint: atomic neighborhoods,
hashing with removal of redundant environments, and folding into a short bit vector. It then measures how often
folding causes bit collisions in a real lead-optimization dataset, and what that does to model explanations.

Entry for the [molab Notebook Competition #3: Cheminformatics Challenge](https://marimo.io/pages/events/notebook-competition-3).

## What is in the notebook

- **Fingerprint anatomy figure**: every atom environment at every radius, wired to the bit it lands on, with
  redundant environments and collisions marked. Molecule, radius, fpSize, ECFP/FCFP invariants and chirality are
  all interactive.
- **Bit inspector**: all environments behind a chosen bit, highlighted on the molecule.
- **Collisions at dataset scale** on the OpenADMET / ExpansionRx training set: share of molecules with collided
  bits, environments per bit, and the colliding pairs that affect the most compounds.
- **Ridge vs LightGBM** on folded and unfolded count fingerprints for eight ADMET endpoints, with exact atom-level
  attribution (weight x count for Ridge, TreeSHAP for LightGBM) and the share of each explanation that is
  "borrowed" from other environments sharing the same bits.
- **Food for thought**: how much of the models' performance is reproduced by eight simple descriptors or by
  atom-type counts alone (ECFP0).

## Running it

Dependencies are declared in the notebook's inline script metadata (PEP 723), so either of these works:

```bash
uvx marimo edit --sandbox ecfp_anatomy.py
```

```bash
uvx marimo run --sandbox ecfp_anatomy.py
```

On [molab](https://molab.marimo.io), add the notebook from its GitHub URL using the new-notebook dropdown.

The dataset is downloaded from Hugging Face when the notebook starts, so an internet connection is needed for the
dataset and model sections. The single-molecule sections work offline.

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

## License

Code: MIT (see [LICENSE](LICENSE)). The ExpansionRx data is not included in this repository; it is
downloaded at runtime and remains under its own CC BY 4.0 license.

## AI disclosure

The original visualization was hand-written for a paper. Converting it into this interactive marimo notebook,
including the dataset and modelling sections, was done with help from Claude (Anthropic).
