import numpy as np
import torch
from pytorch_lightning import LightningDataModule

from torch.utils.data import DataLoader
import re

import os
from .utils.retrieval_two_stage import RetrievalTwoStage_addgtfunction,RetrievalTwoStage_kchengk

Prompt_mode = {
    "retrieval_pairadd":RetrievalTwoStage_addgtfunction,
    "retrieval_paircheng":RetrievalTwoStage_kchengk   
}

class MainDataset(LightningDataModule):
    def __init__(
            self,
            args,
            tokenizer,
            cfg
    ):

        super().__init__()
        self.args = args
        self.cfg = cfg

        self.train_root = cfg["train_root"]
        self.valid_root = cfg["valid_root"]
        self.test_root = cfg['test_root']

        self.tokenizer = tokenizer
        # self.init_tokenizer(tokenizer)
        # .mol_token_id = self.tokenizer.mol_token_id
        # self.passage_token_id = self.tokenizer.passage_token_id
        self.special_token_id = {
            "mol": self.tokenizer.mol_token_id
        }
        # self.fun_token_id = self.tokenizer.fun_token_id

        if self.cfg["mode"] == "pretrain":
            # root, mol_token_id, text_max_len, smiles_prompt1, smiles_prompt2, num_single_mol_token
            self.pretrain_dataset = Prompt_mode[cfg["prompt_model"]](self.train_root, self.special_token_id,self.cfg)
            self.pretrain_dataset.tokenizer = tokenizer
        else:
            self.train_dataset = Prompt_mode[cfg["prompt_model"]](self.train_root, self.special_token_id, self.cfg)
            self.train_dataset.tokenizer = tokenizer
        self.val_dataset = Prompt_mode[cfg["prompt_model"]](self.valid_root, self.special_token_id, self.cfg)
        self.test_dataset = Prompt_mode[cfg["prompt_model"]](self.test_root, self.special_token_id, self.cfg)
        self.val_dataset.tokenizer = tokenizer
        self.test_dataset.tokenizer = tokenizer
        print("self.train_dataset",self.train_root, len(self.train_dataset))
        print("self.val_dataset",self.valid_root, len(self.val_dataset))
        print("self.test_dataset",self.test_root, len(self.test_dataset))

    def train_dataloader(self):
        if self.cfg["mode"] == 'pretrain':
            # print(self.pretrain_dataset)
            loader = DataLoader(
                self.pretrain_dataset,
                batch_size=self.cfg["batch_size"],
                shuffle=True,
                num_workers=self.cfg["num_workers"],
                pin_memory=False,
                drop_last=True,
                persistent_workers=True,
                collate_fn=self.train_dataset.collate_fn)

        elif self.cfg["mode"] == 'ft':
            print("train_dataloader*********************ft**********************")
            loader = DataLoader(
                self.train_dataset,
                batch_size=self.cfg["batch_size"],
                shuffle=True,
                num_workers=self.cfg["num_workers"],
                pin_memory=False,
                drop_last=True,
                persistent_workers=True,
                collate_fn=self.train_dataset.collate_fn
            )
        else:
            raise NotImplementedError
        return loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.cfg["batch_size"],
            shuffle=False,
            num_workers=self.cfg["num_workers"],
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            # ollate_fn=TrainCollater_double(self.tokenizer, self.text_max_len, self.mol_ph_token, self.mol_token_id),
            collate_fn=self.val_dataset.inference_collate_val
        )

        return val_loader
        # return val_loader

    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.cfg["inference_batch_size"],
            shuffle=False,
            num_workers=self.cfg["num_workers"],
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            # collate_fn=InferenceCollater_double(self.tokenizer, self.text_max_len, self.mol_ph_token, self.mol_token_id),
            collate_fn=self.test_dataset.inference_collate_test
        )
        return loader


def count_subdirectories(folder_path):
    try:
        entries = os.listdir(folder_path)

        subdirectories = [entry for entry in entries if os.path.isdir(os.path.join(folder_path, entry))]

        return len(subdirectories)
    except FileNotFoundError:
        print(f"file '{folder_path}' not exsit")
    except Exception as e:
        return -2 