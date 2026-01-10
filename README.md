# g1_hybrid_prior  

## Installation

It's strongly suggested to set up a virtual env; pyenv or uv is preferable but also a conda virtual env is fine if you're more acquainted with the latter. 


### 1. Clone the repository including the submodules

```bash
git clone --recurse-submodules https://github.com/valerio98-lab/g1_hybrid_prior.git
cd g1_hybrid_prior
```

### 2. Install the repo as a package
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
python -m g1_hybrid_prior.data_test --robot g1 --file data/g1_walk_01.csv
```


Finally if you want to visualize the robot trajectories follows the following commands: 

```bash
# Step 1: Set up a Conda virtual environment
conda create -n retarget python=3.10
conda activate retarget

# Step 2: Install dependencies
conda install pinocchio -c conda-forge
pip install numpy rerun-sdk==0.22.0 trimesh

# Step 3: Run the script
python rerun_visualize.py
# run the script with parameters:
# python rerun_visualize.py --file_name dance1_subject2 --robot_type [g1|h1|h1_2]
```
