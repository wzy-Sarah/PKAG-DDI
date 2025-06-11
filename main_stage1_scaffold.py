import torch
from torch import nn
from utils import set_random_seed
from torch.utils.data import Dataset
from prepare_data import remove_element, mol_to_graph_data_obj_simple
import pandas as pd
import argparse
import os
from tools.logging_ import get_root_logger
from torch.utils.data import (DataLoader, RandomSampler)
from torch.optim import Adam, AdamW
from models.mol.gin_model import GNN
from transformers import AutoTokenizer, BioGptModel, AutoModel
from tqdm import tqdm
import time
from torch_geometric.data import Batch
import numpy as np
from collections import defaultdict
import random
import torch.nn.functional as F
import copy
from transformers import OPTForCausalLM
import warnings
import pickle
from models.mol.gin_model import GNN_STM, GNN_graphpred_STM
from sklearn import metrics
from torch.nn import Parameter
import torch.nn.init as init
from sklearn.feature_extraction.text import CountVectorizer
from rdkit import RDLogger
import math
from sklearn.metrics.pairwise import cosine_similarity
import json
import optuna
logger = RDLogger.logger()
logger.setLevel(RDLogger.CRITICAL)
import deepchem as dc
np.set_printoptions(threshold=np.inf)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a scaffold model')
    parser.add_argument('--seednumber', default=42, help='number of seeds')
    parser.add_argument('--mode', type=str, default="train", help='train, valid, test, retrieval')
    parser.add_argument('--split_mode', type=str, default='scaffold', help='scaffold, cold, random')
    parser.add_argument('--fold', type=str, default='0', help='0,1,2')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=40)
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--sample_type', default='random', help='random, inter_cluster, outer_cluster')
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--node_cluster', type=int, default=80)
    parser.add_argument('--weight_decay', type=float, default=0.0004) #6e-05
    parser.add_argument('--mlp_dropout', type=float, default= 0.3)
    parser.add_argument('--mlp_finger_dropout', type=float, default=0.5)
    parser.add_argument('--gnn_dropout', type=float, default=0.5)
    parser.add_argument('--ratio', type=float, default=0.8)

    args = parser.parse_args()
    return args


class AlignDataset(Dataset):
    def __init__(self, args,drug2fingerprint, data_dict,split_mode, mode, stage):
        super(AlignDataset, self).__init__()
        self.all_dict = data_dict
        self.mode = mode
        self.args = args
        self.device = args.device
        self.split_mode = split_mode
        self.fold = args.fold
        self.stage = stage
        self.drug2fingerprint = drug2fingerprint

        if not os.path.exists("data/retrieval/uni_function_list.pkl"):
            self.uni_functions = []
            for k, item in self.all_dict.items():
                # des_id = item["mechanism_des_id"]
                function1, function2 = item["function1"], item["function2"]
                for i in range(len(function1)):
                    f1 = function1[i]
                    f2 = function2[i]
                    if f1 not in self.uni_functions:
                        self.uni_functions.append(f1)
                    if f2 not in self.uni_functions:
                        self.uni_functions.append(f2)
            print("@@@self.uni_functions", len(self.uni_functions))
            with open("data/retrieval/uni_function_list.pkl", 'wb') as f:
                pickle.dump(self.uni_functions, f)
        else:
            with open("data/retrieval/uni_function_list.pkl", 'rb') as f:
                self.uni_functions = pickle.load(f)


        self.all_function_emb = np.eye(len(self.uni_functions))
        self.data_file = self.get_data_file()

        self.get_data(self.data_file)

    def get_data_file(self):
        if self.split_mode == "scaffold":
            data_dir = "./data/MecDDI/scaffold_split/"
            if self.mode == "train":
                data_file = os.path.join(data_dir, "train.txt")
            elif self.mode == "val":
                data_file = os.path.join(data_dir, "val.txt")
            elif self.mode == "test":
                data_file = os.path.join(data_dir, "val.txt")
        elif self.split_mode == "cold":
            data_dir = "./data/MecDDI/cold_split/" + f"fold{self.fold}"
            if self.mode == "train":
                data_file = os.path.join(data_dir, "train.txt")
            elif self.mode == "val":
                data_file = os.path.join(data_dir, "uu.txt")
            elif self.mode == "test":
                data_file = os.path.join(data_dir, "uu.txt")
        elif self.split_mode == "random":
            data_dir = "./data/MecDDI/random_split/"
            if self.mode == "train":
                data_file = os.path.join(data_dir, f"train_seed{self.fold}.txt")
            elif self.mode == "val":
                data_file = os.path.join(data_dir, f"val_seed{self.fold}.txt")
            elif self.mode == "test":
                data_file = os.path.join(data_dir, f"test_seed{self.fold}.txt")
        else:
            assert "Invalid split mode"
        return data_file

    def get_data(self, data_file):
        data_name_list = []
        self.current_ddi = []
        with open(data_file, "r") as f:
            for line in f:
                data_name_list.append(line.strip().split("\n")[0])
        drug2s = pd.read_csv("./data/MecDDI/drug_smiles.csv")
        self.drug2smiles = {a: b for a, b in zip(drug2s["drug_id"], drug2s["smiles"])}

        for i in data_name_list[:200]:
            item = self.all_dict[i]
            drug1, drug2 = item["drug1_name"], item["drug2_name"]
            drug1_function, drug2_function = item["function1"], item["function2"]
            self.current_ddi.append([drug1, drug2, drug1_function, drug2_function,drug1,drug2])
            
    def __getitem__(self, index):
        return self.current_ddi[index]

    def __len__(self):
        return len(self.current_ddi)

    def collate_fn(self, batch):
        drug1, drug2, drug1_function, drug2_function,drug1,drug2 = zip(*batch)
        drug1_list = []
        
        for i in drug1:
            mol = mol_to_graph_data_obj_simple(self.drug2smiles[i])
            drug1_list.append(mol.to(self.device))
        drug1_batch = Batch.from_data_list(drug1_list)

        drug2_list = []
        for i in drug2:
            mol = mol_to_graph_data_obj_simple(self.drug2smiles[i])
            drug2_list.append(mol.to(self.device))
        drug2_batch = Batch.from_data_list(drug2_list)
        
        drug1_finger = []
        for i in drug1:
            ecfp = self.drug2fingerprint[i]
           
            drug1_finger.append(ecfp.squeeze())
        drug1_finger_numpy = np.array(drug1_finger)

        drug2_finger = []
        for i in drug2:
            ecfp = self.drug2fingerprint[i]
            drug2_finger.append(ecfp.squeeze())
        drug2_finger_numpy = np.array(drug2_finger)
        drug1_finger_numpy = torch.tensor(drug1_finger_numpy,dtype=torch.float).to(self.args.device)
        drug2_finger_numpy = torch.tensor(drug2_finger_numpy,dtype=torch.float).to(self.args.device)
        drug1_function_label, drug2_function_label = [], []

        for function in drug1_function:
            drug1_function_label.append(self.uni_functions.index(function[0]))
        for function in drug2_function:
            drug2_function_label.append(self.uni_functions.index(function[0]))


        drug1_function_label = torch.tensor(drug1_function_label, dtype=torch.long).to(self.args.device)
        drug2_function_label = torch.tensor(drug2_function_label, dtype=torch.long).to(self.args.device)
        return (drug1_batch, drug2_batch,drug1_finger_numpy,drug2_finger_numpy,
                drug1_function_label, drug2_function_label,drug1,drug2)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor, mask=None):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)





class MLP(nn.Module):
    def __init__(self, hideen_size_list, input_dim, output_dim, dropout, activation=True, batch_norm=True):
        super(MLP, self).__init__()
        self.model = nn.Sequential()
        hidden_dims = [input_dim] + hideen_size_list + [output_dim]
        for i in range(len(hidden_dims) - 1):
            self.model.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            if i != len(hidden_dims) - 2:
                self.model.append(nn.Dropout(dropout))
                if activation:
                    self.model.append(nn.ReLU())
                if batch_norm:
                    self.model.append(nn.LayerNorm(hidden_dims[i + 1]))

    def forward(self, h):
        return self.model(h)





class Model_stage1(nn.Module):
    def __init__(self, args,uni_functions, logger=None):
        super(Model_stage1, self).__init__()
        # this model used to coarse grained filter the not matched function
        self.args = args
        self.mode = args.mode
        self.uni_functions = uni_functions

        molecule_node_model = GNN_STM(
            num_layer=3, emb_dim=300,
            JK='last', drop_ratio=args.gnn_dropout,
            gnn_type="gin")
        molecule_model = GNN_graphpred_STM(
            num_layer=3, emb_dim=300, JK='last', graph_pooling='mean',
            num_tasks=1, molecule_node_model=molecule_node_model)
        molecule_dim = 300


        self.molecule_model = molecule_model.to(self.args.device)

        self.Mol_Adapter = Mol_Adapter(hidden_dim=300, num_clusters=args.node_cluster)

        self.W_q = Parameter(torch.Tensor(300, 300))
        self.W_k = Parameter(torch.Tensor(300, 300))
        self.W_v_1 = Parameter(torch.Tensor(300, 300))
        self.W_v_2 = Parameter(torch.Tensor(300, 300))
        init.xavier_normal_(self.W_q)
        init.xavier_normal_(self.W_k)
        init.xavier_normal_(self.W_v_1)
        init.xavier_normal_(self.W_v_2)

       
        self.mlp_mol = MLP([600, len(self.uni_functions)], 600, len(self.uni_functions), dropout=self.args.mlp_dropout)
        self.mlp_finger = MLP([1024,300], 1024, 300, dropout=self.args.mlp_finger_dropout)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, batch):
        graphs1, graphs2,finger1, finger2, fun_label1, fun_label2,_,_= batch

        mol1, mask1 = self.molecule_model(graphs1)[0]  
        mol1, _, _ = self.Mol_Adapter(mol1, mask1)  
        mol1_q  = torch.matmul(mol1,self.W_q)
        mol2, mask2 = self.molecule_model(graphs2)[0]
        mol2, _, _ = self.Mol_Adapter(mol2, mask2)  
        mol2_k = torch.matmul(mol2, self.W_k)

        A = mol1_q @ mol2_k.transpose(-1, -2) / (mol2.size(-1) ** 0.5) 

        # for mol1
        A1 = A.softmax(dim=-1)
        mol2_v  = torch.matmul(mol2, self.W_v_1)
        out1 = mol1 + self.args.ratio* (A1 @ mol2_v)
        out1 = torch.mean(out1, dim=1)  # [128,300]
        finger1 = self.mlp_finger(finger1)
        out1 = torch.cat((finger1,out1 ),-1)
        out1 = self.mlp_mol(out1).squeeze()
        loss1 = self.loss(out1, fun_label1)
        # for mol2
        A2 = A.transpose(-1, -2).softmax(dim=-1)
        mol1_v = torch.matmul(mol1, self.W_v_2)
        out2 = mol2 +self.args.ratio*  (A2 @ mol1_v)
        out2 = torch.mean(out2, dim=1)
        finger2 = self.mlp_finger(finger2)
        out2 = torch.cat((finger2,out2 ),-1)
        out2 = self.mlp_mol(out2).squeeze()
        loss2 = self.loss(out2, fun_label2)
        loss = loss1 + loss2
        return loss, out1, out2, fun_label1, fun_label2

    
    def forward_retrieval(self, batch):
        graphs1, graphs2, finger1, finger2, fun_label1, fun_label2, drug_name1, drug_name2 = batch
        mol1, mask1 = self.molecule_model(graphs1)[0]  
        mol1, _, _ = self.Mol_Adapter(mol1, mask1) 
        mol1_q  = torch.matmul(mol1,self.W_q)
        mol2, mask2 = self.molecule_model(graphs2)[0]
        mol2, _, _ = self.Mol_Adapter(mol2, mask2) 
        mol2_k = torch.matmul(mol2, self.W_k)
        A = mol1_q @ mol2_k.transpose(-1, -2) / (mol2.size(-1) ** 0.5) 

        # for mol1
        A1 = A.softmax(dim=-1)
        mol2_v  = torch.matmul(mol2, self.W_v_1)
        out1 = mol1 + self.args.ratio* (A1 @ mol2_v)
        out1 = torch.mean(out1, dim=1) 
        finger1 = self.mlp_finger(finger1)
        out1 = torch.cat((finger1,out1 ),-1)
        out1 = self.mlp_mol(out1).squeeze()
        # for mol2
        A2 = A.transpose(-1, -2).softmax(dim=-1)
        mol1_v = torch.matmul(mol1, self.W_v_2)
        out2 = mol2 +self.args.ratio*  (A2 @ mol1_v)
        out2 = torch.mean(out2, dim=1)
        finger2 = self.mlp_finger(finger2)
        out2 = torch.cat((finger2,out2 ),-1)
        out2 = self.mlp_mol(out2).squeeze()
        return  drug_name1, drug_name2, out1, out2, fun_label1, fun_label2




def do_compute_metrics(probas_pred, target):
    pred = np.argmax(probas_pred, axis=-1)
    assert pred.shape[0] == target.shape[0], f"the shape of pred is {pred.shape}, the shape of target is {target.shape}"
    acc = metrics.accuracy_score(target, pred)
    f1_score = metrics.f1_score(target, pred, average="macro")
    p = metrics.precision_score(target, pred, average="macro", zero_division=0)
    r = metrics.recall_score(target, pred, average="macro", zero_division=0)
    performance = {"acc": acc, "f1": f1_score, "precision": p, "recall": r}
    print("performance:", performance)

    return performance


def evaluate(model, eval_dataloader):
    model.eval()
    nb_eval_steps = 0
    eval_loss = 0
    preds1 = None
    preds2 = None
    gt_emb_ids1 = None
    gt_emb_ids2 = None
    all_preds = None
    all_labels = None
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            
            _, pred_fun1, pred_fun2, gt_id1,  gt_id2 = model(batch)

            nb_eval_steps += 1

            if preds1 is None:
                preds1 = pred_fun1.detach().cpu().numpy()
                preds2 = pred_fun2.detach().cpu().numpy()
                gt_emb_ids1 = gt_id1.detach().cpu().numpy()
                gt_emb_ids2 = gt_id2.detach().cpu().numpy()
       
            else:
                preds1 = np.append(preds1, pred_fun1.detach().cpu().numpy(), axis=0)
                preds2 = np.append(preds2, pred_fun2.detach().cpu().numpy(), axis=0)
                gt_emb_ids1 = np.append(gt_emb_ids1, gt_id1.detach().cpu().numpy(), axis=0)
                gt_emb_ids2 = np.append(gt_emb_ids2, gt_id2.detach().cpu().numpy(), axis=0)
                
    acc1 = do_compute_metrics(preds1, gt_emb_ids1)["acc"]
    acc2 = do_compute_metrics(preds2, gt_emb_ids2)["acc"]

    top_k_num = 2
    sorted_indices_col = np.argsort(-preds1, axis=1)
    top_k_indices1 = sorted_indices_col[:, :top_k_num]
    all_count = 0

    for i in range(sorted_indices_col.shape[0]):
        gt = gt_emb_ids1[i].tolist()
        count = 0
        top_k = list(top_k_indices1[i, :])
        for j in top_k:
            if j == gt:
                count = count + 1
                break
            else:
                continue
        all_count = all_count + count
    print(f"top {top_k_num} acc:", all_count / sorted_indices_col.shape[0])

    top_k_num = 2
    sorted_indices_col2 = np.argsort(-preds2, axis=1)
    top_k_indices2 = sorted_indices_col2[:, :top_k_num]
    all_count = 0

    for i in range(sorted_indices_col2.shape[0]):
        gt = gt_emb_ids2[i].tolist()
        count = 0
        top_k = list(top_k_indices2[i, :])
        for j in top_k:
            if j == gt:
                count = count + 1
                break
            else:
                continue
        all_count = all_count + count
    print(f"top {top_k_num} acc:", all_count / sorted_indices_col2.shape[0])
    
    
    return acc1+acc2


def retrieval(model, eval_dataloader, split_mode, mode, fold):
    def top_k_metric(preds1, gt_emb_ids1, top_k_num):
        sorted_indices_col1 = np.argsort(-preds1, axis=1)
        top_k_indices1 = sorted_indices_col1[:, :top_k_num]
        all_count = 0

        for i in range(sorted_indices_col1.shape[0]):
            gt = gt_emb_ids1[i].tolist()
            count = 0
            top_k = list(top_k_indices1[i, :])
            for j in top_k:
                if j == gt:
                    count = count + 1
                    break
                else:
                    continue
            all_count = all_count + count
        print(f"mol top {top_k_num} acc:", all_count / sorted_indices_col1.shape[0])
        return top_k_indices1

    def top_k_probability(drugname, preds1, gt_emb_ids1, top_k_num):
        sorted_scores, sorted_indices_col1 = torch.sort(torch.tensor(preds1), dim=1, descending=True)        
        top_k_indices1 = sorted_indices_col1[:, :top_k_num] # [samplenum, 5]
        top_k_score = sorted_scores[:,:top_k_num]
        all_count = 0

        for i in range(top_k_indices1.shape[0]):
            gt = gt_emb_ids1[i].tolist()
            count = 0
            top_k = list(top_k_indices1[i, :])
            for j in top_k:
                if j == gt:
                    count = count + 1
                    break
                else:
                    continue
            all_count = all_count + count
        print(f"mol top {top_k_num} acc:", all_count / sorted_indices_col1.shape[0])
        
        return top_k_indices1.detach().numpy(), top_k_score.detach().numpy()

    model.eval()
    eval_loss = 0
    preds1 = None
    preds2 = None
    gt_emb_ids1 = None
    gt_emb_ids2 = None
    drug1_name_list, drug2_name_list = [], []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            # drug_name1, drug_name2, out1, out2, fun_label1, fun_label2
            drug1_name, drug2_name, logit1, logit2, fun_label1, fun_label2 = model.forward_retrieval(batch)
            drug1_name = list(drug1_name)
            drug2_name = list(drug2_name)
            drug1_name_list = drug1_name_list + drug1_name
            drug2_name_list = drug2_name_list + drug2_name

            if preds1 is None:
                preds1 = logit1.detach().cpu().numpy()
                preds2 = logit2.detach().cpu().numpy()
                gt_emb_ids1 = fun_label1.detach().cpu().numpy()
                gt_emb_ids2 = fun_label2.detach().cpu().numpy()
               
            else:
                preds1 = np.append(preds1, logit1.detach().cpu().numpy(), axis=0)
                preds2 = np.append(preds2, logit2.detach().cpu().numpy(), axis=0)
                gt_emb_ids1 = np.append(gt_emb_ids1, fun_label1.detach().cpu().numpy(), axis=0)
                gt_emb_ids2 = np.append(gt_emb_ids2, fun_label2.detach().cpu().numpy(), axis=0)
           
    _ = do_compute_metrics(preds1, gt_emb_ids1)
    _ = do_compute_metrics(preds2, gt_emb_ids2)
    top_k_num = 2
  
    top_k_indices1, top_k_score1 = top_k_probability(drug1_name_list, preds1, gt_emb_ids1, top_k_num)
    top_k_indices2, top_k_score2 = top_k_probability(drug2_name_list, preds2, gt_emb_ids2, top_k_num)
    print("drug1_name_list",len(drug1_name_list),top_k_indices1.shape,top_k_score1.shape)
    print("drug2_name_list",len(drug2_name_list),top_k_indices2.shape,top_k_score2.shape)
    
    data_dict = {}
    for i in range(len(drug1_name_list)):
        drug1 = drug1_name_list[i]
        drug2 = drug2_name_list[i]
        drugpair = drug1 + "&" + drug2
        top_k_ind1 = top_k_indices1[i,:]
        top_k_ind2 = top_k_indices2[i,:]
        top_k_sc1 = top_k_score1[i,:]
        top_k_sc2 = top_k_score2[i,:]
        data_dict[drugpair]=[top_k_ind1, top_k_ind2,top_k_sc1,top_k_sc2]
    

    with open(f"data/retrieval/{split_mode}_{mode}{fold}_retrieval.pkl", 'wb') as file:
        pickle.dump(data_dict, file)
    print(f"save in data/retrieval/{split_mode}_{mode}{fold}_retrieval.pkl")



class BM25:
    def __init__(self, documents, k1=1.2, b=0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.doc_lengths = [len(doc) for doc in documents]
        self.avgdl = sum(self.doc_lengths) / len(self.doc_lengths)
        self.inverted_index = self._build_inverted_index()

    def _build_inverted_index(self):
        index = {}
        for i, doc in enumerate(self.documents):
            for word in set(doc):
                index.setdefault(word, []).append(i)
        return index

    def idf(self, term):
        n_t = len(self.inverted_index.get(term, []))
        return math.log((len(self.documents) - n_t + 0.5) / (n_t + 0.5) + 1)

    def score(self, query, doc_index):
        doc = self.documents[doc_index]
        score = 0
        for term in query:
            if term in doc:
                f = doc.count(term)
                idf = self.idf(term)
                score += idf * ((f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * len(doc) / self.avgdl)))
        return score



def get_gold_function():
    def remove_element(lst, target):
        aaa = [x for x in lst if x != target]
        new = ""
        for id, i in enumerate(aaa):
            if id == 0:
                new = new + i
            else:
                new = new + " " + i
        return new

    with open("./data/MecDDI/all_ddi_addSMIELS_raw.pkl", 'rb') as f:
        ALL_DICT = pickle.load(f)

    new_dict = {}
    for k, v in ALL_DICT.items():
        # print(k)
        drug1, drug2 = v["drug1_name"], v["drug2_name"]
        mechanism_des = v["mechanism_des"]
        mechanism_des_id = v["mechanism_des_id"]
        smiles1, smiles2 = v["SMILES1"], v["SMILES2"]
        drug1_functions, drug2_functions = [],[]

        count = 0
        for s in v.keys():
            if "Mechanism" in s:
                count = count + 1
        # drug1_factor, drug2_factor = "", ""
        for i in range(1, count + 1):
            Mech = v[f"Mechanism_{i}"]
            factor_name1, factor_name2 = Mech["drug1"].strip().split("  "), Mech["drug2"].strip().split("  ")
            # remove the space in factor names
            factor_name1_new = remove_element(factor_name1, '')
            factor_name2_new = remove_element(factor_name2, '')
            drug1_functions.append(factor_name1_new)
            drug2_functions.append(factor_name2_new)
        assert len(drug1_functions)==len(drug2_functions)

        d1 = BM25(drug1_functions)
        d1_scores  = []
        for i in range(len(drug1_functions)):
            d1_scores.append(d1.score(mechanism_des,i))

        d2 = BM25(drug2_functions)
        d2_scores = []
        for i in range(len(drug2_functions)):
            d2_scores.append(d2.score(mechanism_des, i))

        drugpair_scores = [d1_scores[i]+d2_scores[i] for i in range(len(drug1_functions))]

        sorted_pairs1 = sorted(zip(drugpair_scores, drug1_functions), key=lambda x: x[0], reverse=True)
        sorted_function1 = [s for _, s in sorted_pairs1]
        #print(drug1_functions, sorted_function1)

        sorted_pairs2 = sorted(zip(drugpair_scores, drug2_functions), key=lambda x: x[0], reverse=True)
        sorted_function2 = [s for _, s in sorted_pairs2]

        new_dict[k]={'drug1_name':drug1, 'drug2_name':drug2, 'function1':sorted_function1,'function2':sorted_function2}

    return new_dict

#def main(trial:optuna.Trial):
def main(args):
    set_random_seed(args.seednumber)
    data_dict = get_gold_function()
    
    drug2s = pd.read_csv("./data/MecDDI/drug_smiles.csv")
    drug2smiles = {a: b for a, b in zip(drug2s["drug_id"], drug2s["smiles"])}
    drug2fingerprint = {}
    featurizer = dc.feat.CircularFingerprint(size=1024)
    for k, smiles in drug2smiles.items():
        ecfp = featurizer.featurize(smiles)
        
        drug2fingerprint[k]=ecfp
    
    train_dataset = AlignDataset(args,drug2fingerprint, data_dict = data_dict, split_mode=args.split_mode, mode="train", stage="first")
    uni_functions = train_dataset.uni_functions
    val_dataset = AlignDataset(args,drug2fingerprint, data_dict = data_dict, split_mode=args.split_mode, mode="val", stage="first")
    test_dataset = AlignDataset(args,drug2fingerprint, data_dict = data_dict, split_mode=args.split_mode, mode="test", stage="first")
    train_sampler = RandomSampler(train_dataset)
    val_sampler = RandomSampler(val_dataset)
    test_sampler = RandomSampler(test_dataset)

    logger = get_root_logger(log_level='INFO')
    logger.info("The number of train instances: {}".format(len(train_sampler)))
    logger.info("The number of val instances:   {}".format(len(val_sampler)))
    logger.info("The number of test instances:   {}".format(len(test_sampler)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cuda:0")
    args.device = device

    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size,
                                  collate_fn=train_dataset.collate_fn)
    val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=args.batch_size,
                                collate_fn=val_dataset.collate_fn)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size,
                                 collate_fn=test_dataset.collate_fn)

    # get model

    if args.mode == "train":
        best_model = 0
        best_epoch = 0
        model1 = Model_stage1(args,uni_functions, logger)
        model1.to(device)
        # print(model1)
        # for name, param in model1.named_parameters():
        #     if param.requires_grad:
        #         print(name)

        optimizer = AdamW(model1.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        torch.autograd.set_detect_anomaly(True)

        for epoch in range(args.num_epochs):
            batch_step = 0
            batch_loss = 0
            # train_dataloader.sampler.set_epoch(epoch)
            loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
            for step, batch in loop:
                t1 = time.time()
                model1.train()
                optimizer.zero_grad()
                loss, _, _, _, _ = model1(batch)
                # loss = outputs[0]
                loss.backward()
                optimizer.step()

                batch_loss += loss.item()
                batch_step += 1
                loop.set_description(f'Epoch [{epoch}/{args.num_epochs}]')
                loop.set_postfix(loss=loss.item())

            if (epoch + 1) % 5 == 0:
                logger.info(f"epoch is {epoch} || Train batch_loss is {batch_loss / batch_step} \n")
         
                _ =evaluate(model1, val_dataloader)

                # print("time",time.time()-t1)
            if (epoch + 1) % 20 == 0:
                # if torch.distributed.get_rank() == 0:
                torch.save(model1.state_dict(),
                           f'./all_checkpoints/retrieval_epoch{epoch + 1}_split{args.split_mode}_fold{args.fold}.pkl')
        
        acc_all = evaluate(model1, test_dataloader)
        return acc_all

    elif args.mode == "test":
        time1 = time.time()
        model1 = Model_stage1(args,uni_functions, logger).to(args.device)
        model1.load_state_dict(torch.load(
            f'./all_checkpoints/retrieval_epoch{args.num_epochs}_split{args.split_mode}_fold{args.fold}.pkl'))
        evaluate(model1, test_dataloader)
        print("time1",time.time()-time1)
    elif args.mode == "retrieval":
        model1 = Model_stage1(args, uni_functions, logger).to(args.device)
        
        model1.load_state_dict(torch.load(
             f'./all_checkpoints/retrieval_epoch{args.num_epochs}_split{args.split_mode}_fold{args.fold}.pkl'))

        retrieval(model1, test_dataloader, args.split_mode, "test", args.fold)
        retrieval(model1, train_dataloader, args.split_mode, "train", args.fold)
        retrieval(model1, val_dataloader, args.split_mode, "val", args.fold)
        print("saved")



class Mol_Adapter(nn.Module):
    def __init__(self, hidden_dim, num_clusters, residual=False):
        super().__init__()

        self.Q = nn.Parameter(torch.Tensor(1, num_clusters, hidden_dim))
        nn.init.xavier_uniform_(self.Q)

        self.W_Q = nn.Linear(hidden_dim, hidden_dim)
        self.W_K = nn.Linear(hidden_dim, hidden_dim)
        self.W_V = nn.Linear(hidden_dim, hidden_dim)
        self.W_O = nn.Linear(hidden_dim, hidden_dim)

        self.residual = residual

    def forward(self, x, mask):
        K = self.W_K(x)
        V = self.W_V(x)

        # K, mask = to_dense_batch(K, batch)
        # mask: (batch_size, max_num_nodes)
        # V, _ = to_dense_batch(V, batch)

        attn_mask = (~mask).float().unsqueeze(1)
        attn_mask = attn_mask * (-1e9)

        Q = self.Q.tile(K.size(0), 1, 1)
        Q = self.W_Q(Q)

        A = Q @ K.transpose(-1, -2) / (Q.size(-1) ** 0.5)
        A = A + attn_mask
        A = A.softmax(dim=-2)
        
        out = Q + A @ V

        if self.residual:
            out = out + self.W_O(out).relu()
        else:
            out = self.W_O(out).relu()

        return out, A.detach().argmax(dim=-2), mask




if __name__ == "__main__":
    args = parse_args()  # get args
    _=main(args)
  

