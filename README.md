
[![DOI:10.1101/2024.12.05.626885](https://zenodo.org/badge/DOI/10.1101/2024.12.05.626885.svg)](https://doi.org/10.1101/2024.12.05.626885)
[![Requires Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python&logoColor=white)](https://python.org/downloads)


# Biomolecular Emulator (BioEmu)

Biomolecular Emulator (BioEmu for short) is a model that samples from the approximated equilibrium distribution of structures for a protein monomer, given its amino acid sequence.

For more information, see our [preprint](https://www.biorxiv.org/content/10.1101/2024.12.05.626885v1.abstract).

This repository contains inference code and model weights.

## Table of Contents
- [Installation](#installation)
- [Sampling structures](#sampling-structures)
- [Citation](#citation)
- [Get in touch](#get-in-touch)

## Installation

We use git-LFS to store model weights. If you do not already have git-LFS installed, follow the instructions at https://github.com/git-lfs/git-lfs/blob/main/INSTALLING.md, e.g.
```
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git lfs install
git lfs pull
```

Run `setup.sh` to create a conda environment named 'bioemu' with bioemu and its dependencies installed.  `setup.sh` will install and patch [ColabFold](https://github.com/sokrypton/ColabFold), create a conda environment called 'bioemu' with some installed dependencies that pip does not handle, and then pip-install the `bioemu` package inside the conda environment.

## Sampling structures
If you installed `bioemu` in a conda environment named `bioemu` (which is the default if you run `setup.sh` as described above) then you will first need to `conda activate bioemu`.

You can sample structures for a given protein sequence using the script `sample.py`. See `tiny_sample.sh` for an example invocation.

## Citation
If you are using our code or model, please consider citing our work:
```bibtex
@article {BioEmu2024,
	author = {Lewis, Sarah and Hempel, Tim and Jim{\'e}nez-Luna, Jos{\'e} and Gastegger, Michael and Xie, Yu and Foong, Andrew Y. K. and Satorras, Victor Garc{\'\i}a and Abdin, Osama and Veeling, Bastiaan S. and Zaporozhets, Iryna and Chen, Yaoyi and Yang, Soojung and Schneuing, Arne and Nigam, Jigyasa and Barbero, Federico and Stimper, Vincent and Campbell, Andrew and Yim, Jason and Lienen, Marten and Shi, Yu and Zheng, Shuxin and Schulz, Hannes and Munir, Usman and Clementi, Cecilia and No{\'e}, Frank},
	title = {Scalable emulation of protein equilibrium ensembles with generative deep learning},
	year = {2024},
	doi = {10.1101/2024.12.05.626885},
	journal = {bioRxiv}
}
```

## Third-party code
The code in the `openfold` subdirectory is copied from [openfold](https://github.com/aqlaboratory/openfold) with minor modifications. The modifications are described in the relevant source files.
## Get in touch
If you have any questions not covered here, please create an issue or contact the BioEmu team by writing to the corresponding author on our [preprint](https://doi.org/10.1101/2024.12.05.626885).