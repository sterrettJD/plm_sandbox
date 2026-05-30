# PLM sandbox

Building some protein language model toy examples from scratch...

Initially coded on a flight with no wifi, please excuse any slop

## Setup

```
conda create -n plm_sandbox python pip
conda activate plm_sandbox
pip install uv
uv sync
pre-commit install
```

## Basic MLM on a single Pfam

`notebooks/01_mlm_pfama.ipynb`