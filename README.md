# PKAG-DDI
The code of the paper "PKAG-DDI: Pairwise Knowledge-Augmented Language Model for Drug-Drug Interaction Event Text Generation"

The paper can be found in arxiv https://arxiv.org/pdf/2507.19011  because the official version has typo errors.


## Installation environment
You can pip install the following packages:

python==3.10.12

pytorch==2.1.0+cu121

salesforce-lavis

pytorch_lightning==1.9.0

transformers == 4.41.2

torch_geometric==2.6.1

rdkit==2024.03.5

pandas

numpy

peft

scikit-learn

rouge_score

ogb

optuna

deepchem

The **all_checkpoints** and **data (including DDInter2.0 and MecDDI datasets that are used in the paper)** need to be downloaded from [here](https://huggingface.co/datasets/acai233/pkag-ddi/tree/main) . The source of these checkpoints are from MolTC [their checkpoints](https://huggingface.co/chang04/ddi/tree/main) .

## Stage One
Training the PKS and selecting the potential biological function with their relevant scores.

The dataset used in stage one is MecDDI [MecDDI official website](https://mecddi.idrblab.net/). It is a professional DDI dataset that offers medical scientists clear clarification of DDI mechanisms. In our work, we used the DDI and the mechanism information in this dataset.

### Train the PKS

```bash
python main_stage1_random.py --mode train --fold {0, 1, 2}
```

### Test the PKS
```bash
python main_stage1_random.py --mode test --fold {0, 1, 2}
```


### Retrieval using PKS
After training the model, we need to retrieve the training set, validation set, and testing set to get their top-k biological function for the preparation for the next stage.
```bash
python main_stage1_random.py --mode retrieval --fold {0, 1, 2}
```

The output results will be put into ./data/retrieval/

Similarly, you can run main_stage1_cold.py in the same way. The scaffold split does not need to set --fold


## Stage Two

The datasets used in stage two are MecDDI and DDInter2.0 [DDInter2.0 official website](https://ddinter2.scbdd.com/). DDinter2.0 is a comprehensive, professional, and open-access DDI database. To enable comparison and analysis of biological functions, we utilized only the subset of DDIs from the DDInter2.0 dataset that are included in MecDDI and contain biological function information. 
```bash
python prepare_data.py --dataset {mecddi,ddinter} --split_mode {random,cold,scaffold} --fold {0,1,2} --mode {train,val, test}
```


### Train the Generator
For mecddi:
```bash
python main_stage2.py --config configs/final/mecddi_random0_ksquare.json
```


For ddinter:
```bash
python main_stage2.py --config configs/final/ddinter_random0_ksquare.json
```


### Test the Generator
```bash
python main_stage2.py --config config/final/final_ddinter_random0_kchengk.json --mode eval --work_dir work_dir/final_ddinter_random0_kchengk/epoch=*.ckpt
```


### Cite Us
The paper can be found in [here](https://arxiv.org/pdf/2507.19011)
```bibtex
@inproceedings{wang2025pkag,
  title={PKAG-DDI: Pairwise Knowledge-Augmented Language Model for Drug-Drug Interaction Event Text Generation},
  author={Wang, Ziyan and Xiong, Zhankun and Huang, Feng and Zhang, Wen},
  booktitle={Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={10996--11010},
  year={2025}
}
