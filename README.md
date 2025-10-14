
<h1>
<p align="center">
    <img src="assets/emu.png" alt="BioEmu logo" width="300"/>
</p>
</h1>

[![DOI:10.1101/2024.12.05.626885](https://zenodo.org/badge/DOI/10.1101/2024.12.05.626885.svg)](https://doi.org/10.1101/2024.12.05.626885)
[![Requires Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python&logoColor=white)](https://python.org/downloads)

# BioEmu ManyPeptidesMD Baseline

This fork contains an adaptation of the BioEmu codebase for sampling the ManyPeptidesMD evaluation dataset.

The main changes are:
- Modifiying the forcefield to match that used in ManyPeptidesMD.
- Modification of default settings to use suitable numbers of energy evaluations for comparison.

## Installation

```bash
git clone https://github.com/transferable-samplers/bioemu.git
cd bioemu
pip install -e .
pip install -e '.[md]'
```

## Sampling

```bash
sbatch scripts/sample_all.sh
```

## Sidechains + relaxation

Two scripts are provided for the 1e4 and 1e6 energy evaluation budgets employed in "Amortized Sampling with Transferable Normalizing Flows".

In `scripts/sidechain_relax_all_1e4.sh` sidechain packing is performed and only energy minimization.
In `scripts/sidechain_relax_all_1e6.sh` sidechain packing, energy minimization and a short initialization relaxation is performed.

The energy evaluation budget is applied lazily.

# Biomolecular Emulator (BioEmu)

Biomolecular Emulator (BioEmu for short) is a model that samples from the approximated equilibrium distribution of structures for a protein monomer, given its amino acid sequence.

For more information see our <a href="assets/bioemu_paper.pdf" target="_blank">paper</a>, [citation below](#citation).

This repository contains inference code and model weights.

## Table of Contents
- [Installation](#installation)
- [Sampling structures](#sampling-structures)
- [Azure AI Foundry](#azure-ai-foundry)
- [Get in touch](#get-in-touch)
- [Citation](#citation)

## Installation
bioemu is provided as a Linux-only pip-installable package:

```bash
pip install bioemu
```

> [!NOTE]
> The first time `bioemu` is used to sample structures, it will also setup [Colabfold](https://github.com/sokrypton/ColabFold) on a separate virtual environment for MSA and embedding generation. By default this setup uses the `~/.bioemu_colabfold` directory, but if you wish to have this changed please manually set the `BIOEMU_COLABFOLD_DIR` environment variable accordingly before sampling for the first time.


## Sampling structures
You can sample structures for a given protein sequence using the `sample` module. To run a tiny test using the default model parameters and denoising settings:
```
python -m bioemu.sample --sequence GYDPETGTWG --num_samples 10 --output_dir ~/test-chignolin
```

Alternatively, you can use the Python API:

```python
from bioemu.sample import main as sample
sample(sequence='GYDPETGTWG', num_samples=10, output_dir='~/test_chignolin')
```

The model parameters will be automatically downloaded from [huggingface](https://huggingface.co/microsoft/bioemu). A path to a single-sequence FASTA file can also be passed to the `sequence` argument.

Sampling times will depend on sequence length and available infrastructure. The following table gives times for collecting 1000 samples measured on an A100 GPU with 80 GB VRAM for sequences of different lengths (using a `batch_size_100=20` setting in `sample.py`):
 | sequence length | time / min |
 | --------------: | ---------: |
 |             100 |          4 |
 |             300 |         40 |
 |             600 |        150 |

By default, unphysical structures (steric clashes or chain discontinuities) will be filtered out, so you will typically get fewer samples in the output than requested. The difference can be very large if your protein has large disordered regions which are very likely to produce clashes. If you want to get all generated samples in the output, irrespective of whether they are physically valid, use the `--filter_samples=False` argument.


> [!NOTE]
> If you wish to use your own generated MSA instead of the ones retrieved via Colabfold, you can pass an A3M file containing the query sequence as the first row to the `sequence` argument. Additionally, the `msa_host_url` argument can be used to override the default Colabfold MSA query server. See [sample.py](./src/bioemu/sample.py) for more options.

This code only supports sampling structures of monomers. You can try to sample multimers using the [linker trick](https://x.com/ag_smith/status/1417063635000598528), but in our limited experiments, this has not worked well.


## Azure AI Foundry
BioEmu is also available on [Azure AI Foundry](https://ai.azure.com/). See [How to run BioEmu on Azure AI Foundry](AZURE_AI_FOUNDRY.md) for more details.


## Reproducing results from the preprint
You can use this code together with code from [bioemu-benchmarks](https://github.com/microsoft/bioemu-benchmarks) to approximately reproduce results from our [preprint](https://www.biorxiv.org/content/10.1101/2024.12.05.626885v1).

The `bioemu-v1.0` checkpoint contains the model weights used to produce the results in the preprint. Due to simplifications made in the embedding computation and a more efficient sampler, the results obtained with this code are not identical but consistent with the statistics shown in the preprint, i.e., mode coverage and free energy errors averaged over the proteins in a test set. Results for individual proteins may differ. For more details, please check the [BIOEMU_RESULTS.md](https://github.com/microsoft/bioemu-benchmarks/blob/main/bioemu_benchmarks/BIOEMU_RESULTS.md) document on the bioemu-benchmarks repository.


## Side-chain reconstruction and MD-relaxation
BioEmu outputs structures in backbone frame representation. To reconstruct the side-chains, several tools are available. As an example, we interface with [HPacker](https://github.com/gvisani/hpacker) to conduct side-chain reconstruction, and also provide basic tooling for running a short molecular dynamics (MD) equilibration.

> [!WARNING]
> This code is experimental and relies on a [conda-based package manager](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) due to `hpacker` having `conda` as a dependency. Make sure that `conda` is in your `PATH` and that you have CUDA12-compatible drivers before running the following code.

Install optional dependencies:

```bash
pip install bioemu[md]
```

You can compute side-chain reconstructions via the `bioemu.sidechains_relax` module:
```bash
python -m bioemu.sidechain_relax --pdb-path path/to/topology.pdb --xtc-path path/to/samples.xtc
```


> [!NOTE]
> The first time this module is invoked, it will attempt to install `hpacker` and its dependencies into a separate `hpacker` conda environment. If you wish for it to be installed in a different location, please set the `HPACKER_ENV_NAME` environment variable before using this module for the first time.

By default, side-chain reconstruction and local energy minimization are performed (no full MD integration for efficiency reasons).
Note that the runtime of this code scales with the size of the system.
We suggest running this code on a selection of samples rather than the full set.

There are two other options:
- To only run side-chain reconstruction without MD equilibration, add `--no-md-equil`.
- To run a short NVT equilibration (0.1 ns), add `--md-protocol nvt_equil`

To see the full list of options, call `python -m bioemu.sidechain_relax --help`.

The script saves reconstructed all-heavy-atom structures in `samples_sidechain_rec.{pdb,xtc}` and MD-equilibrated structures in `samples_md_equil.{pdb,xtc}` (filename to be altered with `--outname other_name`).

## Third-party code
The code in the `openfold` subdirectory is copied from [openfold](https://github.com/aqlaboratory/openfold) with minor modifications. The modifications are described in the relevant source files.
## Get in touch
If you have any questions not covered here, please create an issue or contact the BioEmu team by writing to the corresponding author on our [preprint](https://doi.org/10.1101/2024.12.05.626885).

## Citation
If you are using our code or model, please cite the following paper:
```bibtex
@article{bioemu2025,
  title={Scalable emulation of protein equilibrium ensembles with generative deep learning},
  author={Lewis, Sarah and Hempel, Tim and Jim{\'e}nez-Luna, Jos{\'e} and Gastegger, Michael and Xie, Yu and Foong, Andrew YK and Satorras, Victor Garc{\'\i}a and Abdin, Osama and Veeling, Bastiaan S and Zaporozhets, Iryna and Chen, Yaoyi and Yang, Soojung and Foster, Adam E. and Schneuing, Arne and Nigam, Jigyasa and Barbero, Federico and Stimper Vincent and  Campbell, Andrew and Yim, Jason and Lienen, Marten and Shi, Yu and Zheng, Shuxin and Schulz, Hannes and Munir, Usman and Sordillo, Roberto and Tomioka, Ryota and Clementi, Cecilia and No{\'e},  Frank},
  journal={Science},
  pages={eadv9817},
  year={2025},
  publisher={American Association for the Advancement of Science},
  doi={10.1126/science.adv9817}
}
```
