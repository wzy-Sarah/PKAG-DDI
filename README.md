# PKAG-DDI
The code of the paper "PKAG-DDI: Pairwise Knowledge-Augmented Language Model for Drug-Drug Interaction Event Text Generation"

## Installation environment
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

all checkpoints are in xxx

download the blip pre-trained model from xxx to all_checkpoints/bert_pretrained, 

download the galactica-1.3b from xxx to all_checkpoints/galactica-1.3b, 

download the pretrained model from xxx to all_checkpoints/stage2

## Stage One
Training the PKS and selecting the potential biological function with their relevant scores.

The dataset used in stage one is MecDDI [MecDDI official website](https://mecddi.idrblab.net/). It is a professional DDI dataset, which could offer the medical scientists clear clarification of DDI mechanisms. In our work, we used the DDI and the mechanism information in this dataset. The dataset we collect is at xxx.

### Train the PKS
python main_stage1_random.py --mode train --fold {0, 1, 2}

### Test the PKS
python main_stage1_random.py --mode test --fold {0, 1, 2}

### Retrieval using PKS
After training the model, we need to retrieve the training set, validation set, and testing set to get their top-k biological function for the preparation for the next stage.
python main_stage1_random.py --mode retrieval --fold {0, 1, 2}
The output results will be put into ./data/retrieval/

Similarly, you can run main_stage1_cold.py in the same way. The scaffold split does not need to set --fold


## Stage Two

The datasets used in stage two are MecDDI and DDInter2.0 [DDInter2.0 official website](https://ddinter2.scbdd.com/). DDinter2.0 is a comprehensive, professional, and open-access DDI database. To enable comparison and analysis of biological functions, we utilized only the subset of DDIs from the DDInter2.0 dataset that are included in MecDDI and contain biological function information. The dataset we collect is at xxx.

python prepare_data.py --dataset {mecddi,ddinter} --split_mode {random,cold,scaffold} --fold {0,1,2} --mode {train,val, test}

### Train the Generator
For mecddi:
python main_stage2.py --config configs/final/mecddi_random0_ksquare.json

For ddinter:
python main_stage2.py --config configs/final/ddinter_random0_ksquare.json

### Test the Generator

python main_stage2.py --config config/final/final_ddinter_random0_kchengk.json --mode eval --work_dir work_dir/final_ddinter_random0_kchengk/epoch=*.ckpt
