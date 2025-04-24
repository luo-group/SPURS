# <div align="center">SPURS</div>
<div align="center">
  <strong>Rewiring protein sequence and structure generative models to enhance protein stability prediction</strong>
</div>

<div align="center">
  <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
  <a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a>
  <br>
  <a href="https://spurs.readthedocs.io/en/latest/"><img alt="Docs" src="https://img.shields.io/badge/DOCS-SPURS-blue"></a>
  <br>
  <a href="https://www.biorxiv.org/content/10.1101/2025.02.13.638154v1"><img alt="Paper" src="https://img.shields.io/badge/Paper-bioRxiv-B31B1B.svg"></a>
</div>

<div align="center">
  <img src="figs/fig1.png" alt="SPURS Model Architecture" width="600"/>
</div>

SPURS is an accurate, rapid, scalable, and generalizable stability predictor. This repository is the official implementation of the paper [Rewiring protein sequence and structure generative models to enhance protein stability prediction](https://www.biorxiv.org/content/10.1101/2025.02.13.638154v1). For detailed documentation, please visit our [documentation site](https://spurs.readthedocs.io/en/latest/).

## 🛠️ Environment

```shell
conda create -n spurs python=3.7 pip
conda activate spurs


pip install -e .
pip install git+https://github.com/facebookresearch/esm.git

pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

## 🔍 Inference
SPURS achieves both efficient and accurate single mutation prediction. Additionally, we have developed an extension module for SPURS that enables multi-mutation prediction. SPURS demonstrates superior performance compared to current leading stability predictors in both single and multi-mutation scenarios.

### Single Mutation Prediction
```python
from spurs.inference import get_SPURS, parse_pdb, get_SPURS_from_hub
# load model from huggingface
# ~ 10s
model, cfg = get_SPURS_from_hub()
pdb_name = 'DOCK1_MOUSE'
pdb_path = './data/inference_example/' + pdb_name + '.pdb'
chain = 'A'
pdb = parse_pdb(pdb_path, pdb_name, chain, cfg)
# ~ 1s
ddg = model(pdb,return_logist=True)
```
The results have been already normalized, so the value in `ddg` for wild-type amino acids are zero.
```python
# wild-type amino acid at position 1
wt_aa = pdb['seq'][0]
ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'
ddg_wt = ddg[0,ALPHABET.index(wt_aa)]
ddg_wt # should be 0
```

For mutation: `W1A`:
```python
mt_aa = 'A'
ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'
ddg_mt = ddg[0,ALPHABET.index(mt_aa)]
ddg_mt # ddg for W1A
```

### Multi-mutation Prediction
```python
from spurs.inference import parse_pdb, get_SPURS_multi_from_hub, parse_pdb_for_mutation
import torch

mut_info_list = [
    ['V2C','P3T'], # multi-mutation 1
    ['W1A','V2Y',], # multi-mutation 2
]
pdb_name = 'DOCK1_MOUSE'
pdb_path = './data/inference_example/' + pdb_name + '.pdb'
chain = 'A'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model, cfg = get_SPURS_multi_from_hub()

pdb = parse_pdb(pdb_path, pdb_name, chain, cfg)
mut_ids, append_tensors = parse_pdb_for_mutation(mut_info_list)
pdb['mut_ids'] = mut_ids
pdb['append_tensors'] = append_tensors.to(device)

ddg = model(pdb)
ddg # ddg[i] for mut_info_list[i]
```

## 🎯 Functional Site Identification
SPURS can assign function scores to each position of the protein with protein sequence and structure as input. An example can be found at [./notebooks/functional_site_identification.ipynb](./notebooks/functional_site_identification.ipynb).

Below is the visualization of the ground truth functional sites(X) and function scores assigned by SPURS for LIM domain in FHL1 (UniProt ID: Q13642; PDB ID: 1X63).
<div align="center">
  <img src="figs/slides_function_score.png" width="500" style="transform: rotate(0deg);" alt="Function Score Visualization"/>
</div>


## 📦 Data
Download `data.tar.gz` from [link](https://www.dropbox.com/scl/fi/uo4e6lvptyy9df5xfulsc/data.tar.gz?rlkey=voi6fxu6ojbzwdk67jlooy8kb&st=4iinnpbc&dl=0).
```shell
tar -xzvf data.tar.gz
```

## 🔄 Reproduce result
### Stability Prediction
Evaluation on the test sets:


```bash
# general usage, only model and test dataset should be specified.
python ./test.py  experiment_path={checkpoint_path} datamodule._target_={dataset_name} data_split=test ckpt_path=best.ckpt mode=predict 
```
while model checkpoints can be selected from [data/checkpoints](./data/checkpoints), and datamodule can be selected from `megascale` and `domainome`.
```bash
## SPURS on Megascale and ten test sets
python ./test.py  experiment_path=data/checkpoints/spurs datamodule._target_=megascale data_split=test ckpt_path=best.ckpt mode=predict 


### ThermoMPNN on Domainome
python ./test.py  experiment_path=data/checkpoints/ThermoMPNN datamodule._target_=domainome data_split=test ckpt_path=best.ckpt mode=predict 
```


Results on Megascale and ten test sets can be processed using [convert.ipynb](./notebooks/convert.ipynb)

### Fitness Prediction

<!-- First change the line 13 of `${CONDA_PREFIX}/lib/python3.7/site-packages/skopt/__init__.py`
```python
from importlib.metadata import version, PackageNotFoundError
```
to 
```python
from importlib_metadata import version, PackageNotFoundError
``` -->



To run fitness prediction
```shell
cd experiments/combining-evolutionary-and-assay-labelled-data
export PROJECT_ROOT=$PWD/../../
python run_proteingym.py
```
This command will use all accessible CPU cores by default. If you want to use a specific range of CPUs, such as CPU0-80, you can use:
```shell
taskset -c 0-80 python run_proteingym.py
```

SPURS-augmented models were built upon the [Augmented models](https://www.nature.com/articles/s41587-021-01146-5) framework (Hsu et al., *Nat biotechnology*, 2022). We adapted the code from the original [GitHub repo](https://github.com/chloechsu/combining-evolutionary-and-assay-labelled-data) (commit `fdaa5bb`) and retained only the necessary files. A [`DDGPredictor`](https://github.com/li-ziang/psnet-release/blob/main/combining-evolutionary-and-assay-labelled-data/src/predictors/esm_predictors.py#L14) is added to introduce predicted ddG into the regression model.



