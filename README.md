# PLM sandbox

Building some protein language model toy examples from scratch...

Initially coded on a flight with no wifi, please excuse any slop

## Setup

```
conda create -n plm_sandbox python pip numpy biopython matplotlib scipy seaborn
conda activate plm_sandbox
pip install torch
```

## Basic MLM on a single Pfam

```
python src/01_mlm_pfama.py
```