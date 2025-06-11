import os
import re
import torch
import random
import pickle
import math
import pandas as pd
import numpy as np
import networkx as nx
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from torch.utils.data import Dataset
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader.dataloader import Collater
from itertools import repeat, chain
from collections import defaultdict
import argparse

def get_gasteiger_partial_charges(mol, n_iter=12):
    """
    Calculates list of gasteiger partial charges for each atom in mol object.
    :param mol: rdkit mol object
    :param n_iter: number of iterations. Default 12
    :return: list of computed partial charges for each atom.
    """
    Chem.rdPartialCharges.ComputeGasteigerCharges(mol, nIter=n_iter,
                                                  throwOnParamFailure=True)
    partial_charges = [float(a.GetProp('_GasteigerCharge')) for a in
                       mol.GetAtoms()]
    return partial_charges


def create_standardized_mol_id(smiles):
    """

    :param smiles:
    :return: inchi
    """
    if check_smiles_validity(smiles):
        # remove stereochemistry
        smiles = AllChem.MolToSmiles(AllChem.MolFromSmiles(smiles),
                                     isomericSmiles=False)
        mol = AllChem.MolFromSmiles(smiles)
        if mol != None:  # to catch weird issue with O=C1O[al]2oc(=O)c3ccc(cn3)c3ccccc3c3cccc(c3)c3ccccc3c3cc(C(F)(F)F)c(cc3o2)-c2ccccc2-c2cccc(c2)-c2ccccc2-c2cccnc21
            if '.' in smiles:  # if multiple species, pick largest molecule
                mol_species_list = split_rdkit_mol_obj(mol)
                largest_mol = get_largest_mol(mol_species_list)
                inchi = AllChem.MolToInchi(largest_mol)
            else:
                inchi = AllChem.MolToInchi(mol)
            return inchi
        else:
            return
    else:
        return


num_atom_type = 120  # including the extra mask tokens
num_chirality_tag = 3
num_bond_type = 6  # including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3

# allowable node and edge features
allowable_features = {
    'possible_atomic_num_list': list(range(1, 119)),
    'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'possible_chirality_list': [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'possible_hybridization_list': [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6],
    'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'possible_bonds': [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs': [  # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ]
}



def mol_to_graph_data_obj_simple(smiles):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses simplified atom and bond features, and represent
    as indices
    :param mol: rdkit mol object
    :return: graph data object with the attributes: x, edge_index, edge_attr
    """
    # atoms
    mol = Chem.MolFromSmiles(smiles)
    num_atom_features = 2  # atom type,  chirality tag
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = [allowable_features['possible_atomic_num_list'].index(
            atom.GetAtomicNum())] + [allowable_features[
                                         'possible_chirality_list'].index(atom.GetChiralTag())]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    num_bond_features = 2  # bond type, bond direction
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [allowable_features['possible_bonds'].index(
                bond.GetBondType())] + [allowable_features[
                'possible_bond_dirs'].index(
                bond.GetBondDir())]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list),
                                 dtype=torch.long)
    else:  # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data


def extract_texts_from_csv(csv_path, id1, id2):
    text1, text2 = None, None

    with open(csv_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(',', 1)  # Split only at the first comma
            if len(parts) == 2:
                current_id, text = parts
                if current_id == id1:
                    text1 = text
                elif current_id == id2:
                    text2 = text
            if text1 is not None and text2 is not None:
                break  # Stop searching if both texts are found

    return text1, text2



import os
from shutil import copy


"""
all_ddi_addSMIELS_raw.pkl
 {'drug1_name', 'drug2_name', 'mechanism_des', 'level', 
 'Mechanism_1': {'drug1': 'Hyperglycemia', 'drug2': 'Antidiabetic agent', 'factor_name1': 'Antidiabetic agents', 'factor_description1'},  
 'Mechanism_2': {'drug1': 'Hyperglycemic effects', 'drug2': 'Antidiabetic agent', 'factor_name2': 'Antidiabetic agents', 'factor_description2'},
 'Management', 'mechanism_des_id': 0, 
 'SMILES1': 'C1=CC=C(C=C1)CCCCOCCCCCCNCC(C2=CC(=C(C=C2)O)CO)O', 
 'SMILES2': 'CC#CCN1C2=C(N=C1N3CCCC(C3)N)N(C(=O)N(C2=O)CC4=NC5=CC=CC=C5C(=N4)C)C'}

Note: The value of 'drug1' and 'drug2' has some space in the string, therefore before we use it, 
 we need to remove the space of the string

"""

"""
all_ddi_addSMIELS.pkl
 {'drug1_name', 'drug2_name', 'mechanism_des', 'mechanism_des_id',
 'function1', 'function2',
 'SMILES1': 'C1=CC=C(C=C1)CCCCOCCCCCCNCC(C2=CC(=C(C=C2)O)CO)O', 
 'SMILES2': 'CC#CCN1C2=C(N=C1N3CCCC(C3)N)N(C(=O)N(C2=O)CC4=NC5=CC=CC=C5C(=N4)C)C'}

"""


def remove_element(lst, target):
    aaa = [x for x in lst if x != target]
    new = ""
    for id, i in enumerate(aaa):
        if id ==0:
            new = new+i
        else:
            new = new+" "+i
    return new

with open("./data/MecDDI/all_ddi_addSMIELS.pkl", 'rb') as f:
    ALL_DICT = pickle.load(f)

def precess_mecddi_rawdata():
    # get the biological function pool
    uni_function=[]
    drug2functio = defaultdict(list)
    drug2smiles = {}
    for k, v in ALL_DICT.items():
        # print(k)
        drug1, drug2 = v["drug1_name"], v["drug2_name"]
        smiles1, smiles2 = v["SMILES1"], v["SMILES2"]
        drug2smiles[drug1]=smiles1
        drug2smiles[drug2]=smiles2
        count = 0
        for s in v.keys():
            if "Mechanism" in s:
                count = count + 1
        # drug1_factor, drug2_factor = "", ""
        for i in range(1, count + 1):
            Mech = v[f"Mechanism_{i}"]
            function_name1, function_name2 = Mech["drug1"].strip().split("  "), Mech["drug2"].strip().split("  ")
            # remove the space in factor names
            function_name1_new = remove_element(function_name1, '')
            function_name2_new = remove_element(function_name2, '')
            if function_name1_new not in uni_function:
                uni_function.append(function_name1_new)
            if function_name2_new not in uni_function:
                uni_function.append(function_name2_new)
            drug2functio[drug1].append(function_name1_new)
            drug2functio[drug2].append(function_name1_new)
    drug2function = defaultdict(list)
    for k, v in drug2functio.items():
        new_factos = []
        for i in v:
            if i not in new_factos:
                new_factos.append(i)
        drug2function[k] = new_factos
    print("biological functions pool has {} functions".format(len(uni_function)))




def process_data(args, input_file, output_dir):
    dataset = args.dataset
    if dataset=="mecddi":
        print("mecddi")
        trainddi_name_list = []
        with open(input_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                trainddi_name_list.append(line.replace("\n", ""))
        for i, ddi in enumerate(trainddi_name_list):
            print(i)
            item = ALL_DICT[ddi]
            drug1_name, drug2_name = item["drug1_name"], item["drug2_name"]
            smiles1, smiles2 = item['SMILES1'], item['SMILES2']
            mechanism_des = item['mechanism_des']
            mech_id = item['mechanism_des_id']
            function1, function2 = item['function1'],item['function2']
    
            os.makedirs(output_dir + "/smiles1/" + str(i))
            os.makedirs(output_dir + "/smiles2/" + str(i))
            os.makedirs(output_dir + "/graph1/" + str(i))
            os.makedirs(output_dir + "/graph2/" + str(i))
            os.makedirs(output_dir + "/text/" + str(i))
            os.makedirs(output_dir + "/drugname1/" + str(i))
            os.makedirs(output_dir + "/drugname2/" + str(i))
            os.makedirs(output_dir + "/function1/" + str(i))
            os.makedirs(output_dir + "/function2/" + str(i))
            data1 = mol_to_graph_data_obj_simple(smiles1)
            torch.save(data1, output_dir + "/graph1/" + str(i) + '/graph_data.pt')
            data2 = mol_to_graph_data_obj_simple(smiles2)
            torch.save(data2, output_dir + "/graph2/" + str(i) + '/graph_data.pt')
            file = open(output_dir + "/smiles1/" + str(i) + "/text.txt", "w")
            file.write(smiles1)
            file.close()
            file = open(output_dir + "/smiles2/" + str(i) + "/text.txt", "w")
            file.write(smiles2)
            file.close()
            test = mechanism_des + '\n'
            file = open(output_dir + "/text/" + str(i) + "/text.txt", "w")
            file.write(test)
            file.close()
    
            # for factor1
            file = open(output_dir + "/function1/" + str(i) + "/text.txt", "w")
            factor_text1 = function1+'\n'
            #print("factor_text1 = " + factor_text1)
            file.write(factor_text1)
            file.close()
    
            # for factor2
            file = open(output_dir + "/function2/" + str(i) + "/text.txt", "w")
            factor_text2 = function2 + "\n"
            file.write(factor_text2)
            file.close()
            #print(f"{drug1_name},{drug2_name}-----{factor_text1}" )
    
    
            file = open(output_dir + "/drugname1/" + str(i) + "/text.txt", "w")
            file.write(drug1_name+ "\n")
            file.close()
    
            file = open(output_dir + "/drugname2/" + str(i) + "/text.txt", "w")
            file.write(drug2_name + "\n")
            file.close()
    else:
        print("ddinter2.0000000000000000")
        trainddi_name_list = []
        with open(input_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                trainddi_name_list.append(line.replace("\n", ""))
        ddinter = pd.read_csv("./data/DDInter2.0/DDInter2_0_mecddi.csv") 
        ddinter_id2item = {a:[b,c,d,e,f] for a,b,c,d,e,f in zip(ddinter["ddiname"], ddinter["drug1"],
                                                             ddinter["drug2"], ddinter["drug1_name"],
                                              ddinter["drug2_name"], ddinter["interaction"])}

        count = 0
        for i, ddi in enumerate(trainddi_name_list):
            print(count)
            
            item = ALL_DICT[ddi]
            drug1_name, drug2_name = item["drug1_name"], item["drug2_name"]
            smiles1, smiles2 = item['SMILES1'], item['SMILES2']
            function1, function2 = item['function1'], item['function2']
            try:
                ddinter_item = ddinter_id2item[ddi]
                mechanism_des = ddinter_item[4]

                os.makedirs(output_dir + "/smiles1/" + str(count))
                os.makedirs(output_dir + "/smiles2/" + str(count))
                os.makedirs(output_dir + "/graph1/" + str(count))
                os.makedirs(output_dir + "/graph2/" + str(count))
                os.makedirs(output_dir + "/text/" + str(count))
                os.makedirs(output_dir + "/drugname1/" + str(count))
                os.makedirs(output_dir + "/drugname2/" + str(count))
                os.makedirs(output_dir + "/function1/" + str(count))
                os.makedirs(output_dir + "/function2/" + str(count))
                data1 = mol_to_graph_data_obj_simple(smiles1)
                torch.save(data1, output_dir + "/graph1/" + str(count) + '/graph_data.pt')
                data2 = mol_to_graph_data_obj_simple(smiles2)
                torch.save(data2, output_dir + "/graph2/" + str(count) + '/graph_data.pt')
                file = open(output_dir + "/smiles1/" + str(count) + "/text.txt", "w")
                file.write(smiles1)
                file.close()
                file = open(output_dir + "/smiles2/" + str(count) + "/text.txt", "w")
                file.write(smiles2)
                file.close()

                test = mechanism_des + '\n'
                print("test = " + test)
                file = open(output_dir + "/text/" + str(count) + "/text.txt", "w")
                file.write(test)
                file.close()
                file = open(output_dir + "/function1/" + str(count) + "/text.txt", "w")
                factor_text1 = function1 + '\n'
                file.write(factor_text1)
                file.close()

                file = open(output_dir + "/function2/" + str(count) + "/text.txt", "w")
                factor_text2 = function2 + "\n"
                file.write(factor_text2)
                file.close()

                file = open(output_dir + "/drugname1/" + str(count) + "/text.txt", "w")
                file.write(drug1_name + "\n")
                file.close()

                file = open(output_dir + "/drugname2/" + str(count) + "/text.txt", "w")
                #print("drug2_name", drug2_name)
                file.write(drug2_name + "\n")
                file.close()
                count = count+1
            except:
                continue


        #print(i)

def count_subdirectories(folder_path):
    try:
        entries = os.listdir(folder_path)
        subdirectories = [entry for entry in entries if os.path.isdir(os.path.join(folder_path, entry))]

        return len(subdirectories)
    except FileNotFoundError:
        return -1 
    except Exception as e:
        return -2 





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a graph model')
    parser.add_argument('--dataset', help='the type of dataset, mecddi, ddinter')
    parser.add_argument('--split_mode', help='random,cold or scaffold')
    parser.add_argument('--mode', help='train, val, or test')
    parser.add_argument('--fold', help='0,1,2')
    args = parser.parse_args()
    if args.dataset == "mecddi":
        dataset = "MecDDI"
        if args.split_mode =="random":
            input_file = f"./data/{dataset}/random_split/{args.mode}_seed{args.fold}.txt"
            output_dir = f"./data/{dataset}_inputdata/{args.mode}_{args.split_mode}_split{args.fold}"
        elif args.split_mode =="cold":
            if args.mode=="train":
                input_file = f"./data/{dataset}/cold_split/fold{args.fold}/train.txt"
                output_dir = f"./data/{dataset}_inputdata/{args.mode}_{args.split_mode}_split{args.fold}"
            elif args.mode=="val":
                input_file = f"./data/{dataset}/cold_split/fold{args.fold}/su.txt"
                output_dir = f"./data/{dataset}_inputdata/{args.mode}_{args.split_mode}_split{args.fold}"
            elif args.mode=="test":
                input_file = f"./data/{dataset}/cold_split/fold{args.fold}/uu.txt"
                output_dir = f"./data/{dataset}_inputdata/{args.mode}_{args.split_mode}_split{args.fold}"
            else:
                print("error")
        elif args.split_mode=="scaffold":
            if args.mode=="train":
                input_file = f"./data/{dataset}/scaffold_split/train.txt"
                output_dir = f"./data/{dataset}_inputdata/{args.mode}_{args.split_mode}_split{args.fold}"
            else:
                input_file = f"./data/{dataset}/scaffold_split/val.txt" #unseen and unseen
                output_dir = f"./data/{dataset}_inputdata/{args.mode}_{args.split_mode}_split{args.fold}"
        else:
            print("error")
    elif args.dataset=="ddinter":
        if args.split_mode =="random":
            input_file = f"./data/MecDDI/random_split/{args.mode}_seed{args.fold}.txt"
            output_dir = f"./data/ddinter_inputdata/{args.mode}_{args.split_mode}_split{args.fold}"
        elif args.split_mode =="cold":
            if args.mode=="train":
                input_file = f"./data/MecDDI/cold_split/fold{args.fold}/train.txt"
                output_dir = f"./data/ddinter_inputdata/{args.mode}_{args.split_mode}_split{args.fold}"
            elif args.mode=="val":
                input_file = f"./data/MecDDI/cold_split/fold{args.fold}/su.txt"
                output_dir = f"./data/ddinter_inputdata/{args.mode}_{args.split_mode}_split{args.fold}"
            elif args.mode=="test":
                input_file = f"./data/MecDDI/cold_split/fold{args.fold}/uu.txt"
                output_dir = f"./data/ddinter_inputdata/{args.mode}_{args.split_mode}_split{args.fold}"
            else:
                print("error")
        elif args.split_mode=="scaffold":
            if args.mode=="train":
                input_file = f"./data/MecDDI/scaffold_split/train.txt"
                output_dir = f"./data/ddinter_inputdata/{args.mode}_{args.split_mode}_split{args.fold}"
            else:
                input_file = f"./data/MecDDI/scaffold_split/val.txt" #unseen and unseen
                output_dir = f"./data/ddinter_inputdata/{args.mode}_{args.split_mode}_split{args.fold}"
        else:
            print("error")
   
    process_data(args, input_file =input_file, output_dir=output_dir)



