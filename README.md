# g1_hybrid_prior  

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/valerio98-lab/g1_hybrid_prior.git
cd g1_hybrid_prior
```

```bash
pip install -e .
```

After installation, the main command becomes available:

```bash
g1-hybrid-prior --robot g1 --file path/to/log.csv
```

Example: 

```bash
g1-hybrid-prior \
    --robot g1 \
    --file data/g1_walk_01.csv \
    --n_frames 3
```

Or you may also run it via Python module execution: 

```bash
python -m g1_hybrid_prior.cli --robot g1 --file data/g1_walk_01.csv
```

