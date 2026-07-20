# PLM sandbox

Building some protein language model and folding/structural toy examples from scratch...

This started as a project on a flight with no wifi, please excuse any human-derived slop

All of these projects have been completed without code generation via AI coding tools, the old fashioned way to support learning/skill development.


## Setup

```
conda create -n plm_sandbox python pip
conda activate plm_sandbox
pip install uv
uv sync
uv run pre-commit install
```

## 1. Basic MLM on a single Pfam

This project trains a basic BERT-style masked language model on a single PFAM.
Source code includes protein tokenizing and MLM architecture.
An example is executed in `notebooks/01_mlm_pfama.ipynb`

## 2. Global vs pocket RMSD calculation

This project involves writing a PDB file parser, pocket detector, Kabsch superposition, and RMSD calculation.
