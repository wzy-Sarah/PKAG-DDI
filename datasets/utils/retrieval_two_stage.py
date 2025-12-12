from torch_geometric.data import Dataset
import os
from torch_geometric.loader.dataloader import Collater
import numpy as np
import torch
import re
from transformers import AutoTokenizer, BioGptModel, AutoModel
import pickle
from torch_geometric.data import Batch
import pandas as pd
from itertools import chain


class RetrievalTwoStage_addgtfunction(Dataset):

    def __init__(self, root, mol_token_id, cfg):
        super(RetrievalTwoStage_addgtfunction, self).__init__(root)
        
        self.root = root  # the input data root
        self.text_max_len = cfg["text_max_len"]
        self.tokenizer = None
        self.mol_prompt1 = cfg["mol_prompt1"]
        self.mol_prompt2 = cfg["mol_prompt2"]
        self.mol_ph = '<mol>' * cfg["num_single_mol_token"]
        self.mol_token_id = mol_token_id["mol"]
        self.collater = Collater([], [])
        self.retrieval_number = cfg["retrieval_function_number"]
        self.add_gt = cfg["add_gt"]

        ### retrieval index
        all_function_list_file = "data/retrieval/uni_function_list.pkl"
        with open(all_function_list_file, 'rb') as f:
            self.uni_function_list = pickle.load(f)

        ### load retrieval
        file_name = os.path.basename(os.path.normpath(self.root)) # train_random_split0
        split_mode = file_name.split("_")[1] #random
        self.mode = file_name.split("_")[0] #train
        fold = file_name[-1]
        retrie_file = split_mode + "_" + self.mode + fold + "_retrieval" + ".pkl" # random_train0_retrieval.csv
        retrie_file = os.path.join("data/retrieval", retrie_file)
        with open(retrie_file, 'rb') as file:
            self.twodrug2topk = pickle.load(file)



    def get(self, index):
        return self.__getitem__(index)

    def len(self):
        return len(self)

    def __len__(self):
        if 'train' in self.root:
            return int(count_subdirectories(self.root + "text/"))
        else:
            return int(count_subdirectories(self.root + "text/"))

    def __getitem__(self, index):
        drug1_name_list = os.listdir(self.root + 'drugname1/' + str(index) + '/') # text.txt
        drug2_name_list = os.listdir(self.root + 'drugname2/' + str(index) + '/')
        graph1_name_list = os.listdir(self.root + 'graph1/' + str(index) + '/')
        graph2_name_list = os.listdir(self.root + 'graph2/' + str(index) + '/')
        text_name_list = os.listdir(self.root + 'text/' + str(index) + '/')
        if self.mode=="train" and self.add_gt:
            function1_name_list = os.listdir(self.root + 'function1/' + str(index) + '/')
            function2_name_list = os.listdir(self.root + 'function2/' + str(index) + '/')
        smiles1_name_list = os.listdir(self.root+'smiles1/'+str(index)+'/')
        smiles2_name_list = os.listdir(self.root+'smiles2/'+str(index)+'/')

        

        # load drug 1
        drug_path1 = os.path.join(self.root, 'drugname1/' + str(index) + '/', drug1_name_list[0])
        with open(drug_path1, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            assert len(lines) == 1
            drug1_name = lines[0].strip('\n').strip()
        graph_path = os.path.join(self.root, 'graph1/' + str(index) + '/', graph1_name_list[0])
        data_graph1 = torch.load(graph_path)
        smiles_path1 = os.path.join(self.root, 'smiles1/'+str(index)+'/', smiles1_name_list[0])
        with open(smiles_path1, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            assert len(lines) == 1
            smiles1 = lines[0].strip()
        if self.mode=="train" and self.add_gt:
            function1_path = os.path.join(self.root, 'function1/' + str(index) + '/', function1_name_list[0])
            with open(function1_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                assert len(lines) == 1
                function1 = lines[0].strip('\n').strip()

        # load drug 2
        drug_path2 = os.path.join(self.root, 'drugname2/' + str(index) + '/', drug2_name_list[0])
        with open(drug_path2, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            assert len(lines) == 1
            drug2_name = lines[0].strip('\n').strip()
        graph_path = os.path.join(self.root, 'graph2/' + str(index) + '/', graph2_name_list[0])
        data_graph2 = torch.load(graph_path)
        smiles_path2 = os.path.join(self.root, 'smiles2/'+str(index)+'/', smiles2_name_list[0])
        with open(smiles_path2, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            assert len(lines) == 1
            smiles2 = lines[0].strip()  
        if self.mode =="train" and self.add_gt:
            function2_path = os.path.join(self.root, 'function2/' + str(index) + '/', function2_name_list[0])
            with open(function2_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                assert len(lines) == 1
                function2 = lines[0].strip('\n').strip()


        two_drug_name = drug1_name+"&"+drug2_name
        retrieval_infor = self.twodrug2topk[two_drug_name]
        drug1_retrieval = retrieval_infor[0][:self.retrieval_number] # first is the function index, second is function prob
        drug2_retrieval = retrieval_infor[1][:self.retrieval_number]
        top_k_score1 = retrieval_infor[2][:self.retrieval_number]
        top_k_score2 = retrieval_infor[3][:self.retrieval_number]


        assert len(drug1_retrieval) == len(drug2_retrieval),f"retrieval func1:{len(drug1_retrieval)}; retrieval func2:{len(drug2_retrieval)}"
        # The input of two drugs,contain smiles, mol, function
        top_k_CoT_prompt = []
        for id, func1 in enumerate(drug1_retrieval):
            func2 = drug2_retrieval[id]
            mol_prompt1 = self.mol_prompt1.format(smiles1[:128], self.uni_function_list[func1] )
            mol_prompt2 = self.mol_prompt2.format(smiles2[:128], self.uni_function_list[func2] )
            mol_prompt = '<s>' + mol_prompt1 + '</s>' + ' <s>' + mol_prompt2 + '</s>'
            top_k_CoT_prompt.append(mol_prompt)
        if self.mode=="train" and self.add_gt :
            top_k_CoT_prompt.append('<s>' + self.mol_prompt1.format(smiles1[:128], function1 ) + '</s>' + ' <s>' + self.mol_prompt2.format(smiles2[:128], function2 ) + '</s>')

        inputs = top_k_CoT_prompt

       

        # load ddie description
        text_path = os.path.join(self.root, 'text/' + str(index) + '/', text_name_list[0])
        outputs = []
        count = 0
        for line in open(text_path, 'r', encoding='utf-8'):
            count += 1
            ddie = line.strip('\n')
            #print(ddie+'.')
            outputs.append(str(ddie) + '.')
            if count > 100:
                break


        return data_graph1, data_graph2, inputs, drug1_name, drug2_name, outputs, top_k_score1, top_k_score2


    def collate_fn(self, batch):
        graphs1, graphs2, inputs, drug1_name, drug2_name, outputs, top_k_scores1, top_k_scores2 = zip(*batch)

        graphs1 =  [item for item in graphs1 for _ in range(len(inputs[0]))]
        graphs2 =  [item for item in graphs2 for _ in range(len(inputs[0]))]

        graphs1 = self.collater(graphs1)
        graphs2 = self.collater(graphs2)
        
        
        ## deal with prompt
        all_inputs = list(chain(*inputs))
        new_input_prompt = [smiles_handler(p, self.mol_ph)[0] for p in all_inputs]
       
       
        smiles_prompt_tokens = self.tokenizer(text=new_input_prompt,
                                              truncation=False,
                                              padding='longest',
                                              add_special_tokens=True,
                                              return_tensors='pt',
                                              return_attention_mask=True)
        #print("aaasmiles_prompt_tokens",len(smiles_prompt_tokens))

        is_mol_token = smiles_prompt_tokens.input_ids == self.mol_token_id  # the indexs that the input_ids has the element that == mol_token_id
        
        smiles_prompt_tokens['is_mol_token'] = is_mol_token



        ### get the retrieval scores
        topk_scores_numpy1 = torch.tensor(np.array(top_k_scores1)) # [batchsize,topk]
        topk_scores_numpy2 = torch.tensor(np.array(top_k_scores2))

        

        outputss = []
        for item in outputs:
            outputss.append(item[0] + " <EOS>")
    
        outputss = [item for item in outputss for _ in range(len(inputs[0]))]
        
        output_tokens = self.tokenizer(text=outputss,
                                       truncation=True,
                                       padding='longest',
                                       add_special_tokens=True,
                                       #max_length=self.text_max_len,
                                       return_tensors='pt',
                                       return_attention_mask=True)
        
       
        return graphs1, graphs2, smiles_prompt_tokens, drug1_name, drug2_name, output_tokens,topk_scores_numpy1, topk_scores_numpy2

    def inference_collate_val(self, batch):
        # print(batch)
        graphs1, graphs2, inputs, drug1_name, drug2_name, outputs, top_k_scores1, top_k_scores2 = zip(*batch)
        graphs1 = self.collater(graphs1)
        graphs2 = self.collater(graphs2)
        
        
        all_inputs = []
        for item in inputs:
            all_inputs.append(item[0])
        new_input_prompt = [smiles_handler(p, self.mol_ph)[0] for p in all_inputs]
        


        ## deal with prompt
        smiles_prompt_tokens = self.tokenizer(new_input_prompt,
                                              return_tensors='pt',
                                              #    max_length=self.text_max_len,
                                              padding='longest',
                                              truncation=False,
                                              return_attention_mask=True)
        is_mol_token = smiles_prompt_tokens.input_ids == self.mol_token_id
        smiles_prompt_tokens['is_mol_token'] = is_mol_token

        
        outputss = []
        gt_outputs = []
        #aaa = []
        for item in outputs:
            outputss.append(item[0] + "<EOS>")
            gt_outputs.append(item[0])

        output_tokens = self.tokenizer(text=outputss,
                                       truncation=True,
                                       padding='longest',
                                       add_special_tokens=False,
                                       max_length=self.text_max_len,
                                       return_tensors='pt',
                                       return_attention_mask=True)

        return graphs1, graphs2, smiles_prompt_tokens, drug1_name, drug2_name, output_tokens ,topk_scores_numpy1,topk_scores_numpy2, top_k_CoT_prompt, gt_outputs
       
    def inference_collate_test(self, batch):
        graphs1, graphs2, inputs, drug1_name, drug2_name, outputs, top_k_scores1, top_k_scores2 = zip(*batch)
        graphs1 = self.collater(graphs1)
        graphs2 = self.collater(graphs2)
        #all_inputs = list(chain(*inputs))
        all_inputs = []
        for item in inputs:
            all_inputs.append(item[0])
        
        smiles_prompt = [smiles_handler(p, self.mol_ph)[0] for p in all_inputs]

        ## deal with prompt
        smiles_prompt_tokens = self.tokenizer(smiles_prompt,
                                              return_tensors='pt',
                                              #    max_length=self.text_max_len,
                                              padding='longest',
                                              truncation=False,
                                              return_attention_mask=True)

        is_mol_token = smiles_prompt_tokens.input_ids == self.mol_token_id
        smiles_prompt_tokens['is_mol_token'] = is_mol_token
        
        outputss = []
        for item in outputs:
            outputss.append(item[0])

        #return graphs1, graphs2, smiles_prompt_tokens, None, None, None,None,None,  top_k_CoT_prompt, outputss
        return graphs1, graphs2, smiles_prompt_tokens, drug1_name, drug2_name, None,None,None,  None, outputss

    def tokenizer_text(self, text):
        sentence_token = self.tokenizer(text=text,
                                        truncation=True,
                                        padding='max_length',
                                        add_special_tokens=True,
                                        max_length=self.text_max_len,
                                        return_tensors='pt',
                                        return_attention_mask=True)
        return sentence_token








class RetrievalTwoStage_kchengk(Dataset):
    # no add gt function

    def __init__(self, root, mol_token_id, cfg):
        super(RetrievalTwoStage_kchengk, self).__init__(root)
        # text_max_len, smiles_prompt1, smiles_prompt2, num_single_mol_token
        # this is output CoT
        self.root = root  # the input data root
        self.text_max_len = cfg["text_max_len"]
        self.tokenizer = None
        self.mol_prompt1 = cfg["mol_prompt1"]
        self.mol_prompt2 = cfg["mol_prompt2"]
        self.mol_ph = '<mol>' * cfg["num_single_mol_token"]
        self.mol_token_id = mol_token_id["mol"]
        self.collater = Collater([], [])
        self.retrieval_number = cfg["retrieval_function_number"]
        #self.smiles_prompt = 'The SMILES of this molecule is [START_I_SMILES]{}[END_I_SMILES]. '

        ### retrieval index
        all_function_list_file = "data/retrieval/uni_function_list.pkl"
        with open(all_function_list_file, 'rb') as f:
            self.uni_function_list = pickle.load(f)

        ### load retrieval
        file_name = os.path.basename(os.path.normpath(self.root)) # train_random_split0
        split_mode = file_name.split("_")[1] #random
        print("split_mode",split_mode)
        mode = file_name.split("_")[0] #train
        print("mode",mode)
        fold = file_name[-1]
        retrie_file = split_mode + "_" + mode + fold + "_retrieval" + ".pkl" # random_train0_retrieval.csv
        retrie_file = os.path.join("data/retrieval", retrie_file)
        print("retrie_file",retrie_file)
        with open(retrie_file, 'rb') as file:
            self.twodrug2topk = pickle.load(file)



    def get(self, index):
        return self.__getitem__(index)

    def len(self):
        return len(self)

    def __len__(self):
        if 'train' in self.root:     
            return int(count_subdirectories(self.root + "text/"))
        else:
            return int(count_subdirectories(self.root + "text/"))
            

    def __getitem__(self, index):
        drug1_name_list = os.listdir(self.root + 'drugname1/' + str(index) + '/') # text.txt
        drug2_name_list = os.listdir(self.root + 'drugname2/' + str(index) + '/')
        graph1_name_list = os.listdir(self.root + 'graph1/' + str(index) + '/')
        graph2_name_list = os.listdir(self.root + 'graph2/' + str(index) + '/')
        text_name_list = os.listdir(self.root + 'text/' + str(index) + '/')
        function1_name_list = os.listdir(self.root + 'function1/' + str(index) + '/')
        function2_name_list = os.listdir(self.root + 'function2/' + str(index) + '/')
        smiles1_name_list = os.listdir(self.root+'smiles1/'+str(index)+'/')
        smiles2_name_list = os.listdir(self.root+'smiles2/'+str(index)+'/')

        

        # load drug 1
        drug_path1 = os.path.join(self.root, 'drugname1/' + str(index) + '/', drug1_name_list[0])
        with open(drug_path1, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            assert len(lines) == 1
            drug1_name = lines[0].strip('\n').strip()
        graph_path = os.path.join(self.root, 'graph1/' + str(index) + '/', graph1_name_list[0])
        data_graph1 = torch.load(graph_path)
        smiles_path1 = os.path.join(self.root, 'smiles1/'+str(index)+'/', smiles1_name_list[0])
        with open(smiles_path1, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            assert len(lines) == 1
            smiles1 = lines[0].strip()
        function1_path = os.path.join(self.root, 'function1/' + str(index) + '/', function1_name_list[0])
        with open(function1_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            assert len(lines) == 1
            function1 = lines[0].strip('\n').strip()

        # load drug 2
        drug_path2 = os.path.join(self.root, 'drugname2/' + str(index) + '/', drug2_name_list[0])
        with open(drug_path2, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            assert len(lines) == 1
            drug2_name = lines[0].strip('\n').strip()
        graph_path = os.path.join(self.root, 'graph2/' + str(index) + '/', graph2_name_list[0])
        data_graph2 = torch.load(graph_path)
        smiles_path2 = os.path.join(self.root, 'smiles2/'+str(index)+'/', smiles2_name_list[0])
        with open(smiles_path2, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            assert len(lines) == 1
            smiles2 = lines[0].strip()  
        function2_path = os.path.join(self.root, 'function2/' + str(index) + '/', function2_name_list[0])
        with open(function2_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            assert len(lines) == 1
            function2 = lines[0].strip('\n').strip()


        two_drug_name = drug1_name+"&"+drug2_name
        retrieval_infor = self.twodrug2topk[two_drug_name]
        #print("retrieval_infor",retrieval_infor)
        drug1_retrieval = retrieval_infor[0][:self.retrieval_number] # first is the function index, second is function prob
        drug2_retrieval = retrieval_infor[1][:self.retrieval_number]
        top_k_score1 = retrieval_infor[2][:self.retrieval_number]
        top_k_score2 = retrieval_infor[3][:self.retrieval_number]


        assert len(drug1_retrieval) == len(drug2_retrieval),f"retrieval func1:{len(drug1_retrieval)}; retrieval func2:{len(drug2_retrieval)}"


        # The input of two drugs,contain smiles, mol, function
        top_k_CoT_prompt = []
        
        for id, func1 in enumerate(drug1_retrieval):
            for func2 in drug2_retrieval:
                mol_prompt1 = self.mol_prompt1.format(smiles1[:128], self.uni_function_list[func1] )
                mol_prompt2 = self.mol_prompt2.format(smiles2[:128], self.uni_function_list[func2] )
                mol_prompt = '<s>' + mol_prompt1 + '</s>' + ' <s>' + mol_prompt2 + '</s>'
                top_k_CoT_prompt.append(mol_prompt)
        
        
        inputs = top_k_CoT_prompt      

        # load ddie description
        text_path = os.path.join(self.root, 'text/' + str(index) + '/', text_name_list[0])
        outputs = []
        count = 0
        for line in open(text_path, 'r', encoding='utf-8'):
            count += 1
            ddie = line.strip('\n')
            #print(ddie+'.')
            outputs.append(str(ddie) + '.')
            if count > 100:
                break
        
        return data_graph1, data_graph2, inputs, drug1_name, drug2_name, outputs, top_k_score1, top_k_score2

    def collate_fn(self, batch):
        graphs1, graphs2, inputs, drug1_name, drug2_name, outputs, top_k_scores1, top_k_scores2 = zip(*batch)

        graphs1 =  [item for item in graphs1 for _ in range(len(inputs[0]))]
        graphs2 =  [item for item in graphs2 for _ in range(len(inputs[0]))]

        graphs1 = self.collater(graphs1)
        graphs2 = self.collater(graphs2)
        
        
        ## deal with prompt
        all_inputs = list(chain(*inputs))
        new_input_prompt = [smiles_handler(p, self.mol_ph)[0] for p in all_inputs]
       
       
        smiles_prompt_tokens = self.tokenizer(text=new_input_prompt,
                                              truncation=False,
                                              padding='longest',
                                              add_special_tokens=True,
                                              return_tensors='pt',
                                              return_attention_mask=True)
        #print("aaasmiles_prompt_tokens",len(smiles_prompt_tokens))

        is_mol_token = smiles_prompt_tokens.input_ids == self.mol_token_id  # the indexs that the input_ids has the element that == mol_token_id
        #print("is_mol_token", is_mol_token)
        smiles_prompt_tokens['is_mol_token'] = is_mol_token


        ### get the retrieval scores
        topk_scores_numpy1 = torch.tensor(np.array(top_k_scores1)) # [batchsize,topk]
        topk_scores_numpy2 = torch.tensor(np.array(top_k_scores2))

        

        outputss = []
        for item in outputs:
            outputss.append(item[0] + " <EOS>")
    
        #outputss = outputss * len(top_k_CoT_prompt[0])
        outputss = [item for item in outputss for _ in range(len(inputs[0]))]
        
        output_tokens = self.tokenizer(text=outputss,
                                       truncation=True,
                                       padding='longest',
                                       add_special_tokens=True,
                                       #max_length=self.text_max_len,
                                       return_tensors='pt',
                                       return_attention_mask=True)
        return graphs1, graphs2, smiles_prompt_tokens, drug1_name, drug2_name, output_tokens,topk_scores_numpy1, topk_scores_numpy2

    def inference_collate_val(self, batch):
        # print(batch)
        graphs1, graphs2, inputs, drug1_name, drug2_name, outputs, top_k_scores1, top_k_scores2 = zip(*batch)
        graphs1 =  [item for item in graphs1 for _ in range(len(inputs[0]))]
        graphs2 =  [item for item in graphs2 for _ in range(len(inputs[0]))]
        graphs1 = self.collater(graphs1)
        graphs2 = self.collater(graphs2)
        
        #print("graphs2",graphs2,len(graphs2))
        all_inputs = list(chain(*inputs))
        #print("all_inputs",len(all_inputs))
        new_input_prompt = [smiles_handler(p, self.mol_ph)[0] for p in all_inputs]
        


        ## deal with prompt
        smiles_prompt_tokens = self.tokenizer(new_input_prompt,
                                              return_tensors='pt',
                                              #    max_length=self.text_max_len,
                                              padding='longest',
                                              truncation=False,
                                              return_attention_mask=True)
        is_mol_token = smiles_prompt_tokens.input_ids == self.mol_token_id
        smiles_prompt_tokens['is_mol_token'] = is_mol_token

 
        ### get the retrieval scores
        topk_scores_numpy1 = torch.tensor(np.array(top_k_scores1)) # [batchsize,topk]
        topk_scores_numpy2 = torch.tensor(np.array(top_k_scores2))
        
        
        outputss = []
        gt_outputs = []
        #aaa = []
        for item in outputs:
            outputss.append(item[0] + "<EOS>")
            gt_outputs.append(item[0])
            #aaa.append("<EOS>")

        outputss = [item for item in outputss for _ in range(len(inputs[0]))]
        output_tokens = self.tokenizer(text=outputss,
                                       truncation=True,
                                       padding='longest',
                                       add_special_tokens=False,
                                       max_length=self.text_max_len,
                                       return_tensors='pt',
                                       return_attention_mask=True)

        return graphs1, graphs2, smiles_prompt_tokens, None, None, output_tokens ,topk_scores_numpy1,topk_scores_numpy2, None, gt_outputs

    def inference_collate_test(self, batch):
        graphs1, graphs2, inputs, drug1_name, drug2_name, outputs, top_k_scores1, top_k_scores2 = zip(*batch)
        graphs1 =  [item for item in graphs1 for _ in range(len(inputs[0]))]
        graphs2 =  [item for item in graphs2 for _ in range(len(inputs[0]))]
        graphs1 = self.collater(graphs1)
        graphs2 = self.collater(graphs2)
        
        all_inputs = list(chain(*inputs))
        new_input_prompt = [smiles_handler(p, self.mol_ph)[0] for p in all_inputs]
       

        ## deal with prompt
        smiles_prompt_tokens = self.tokenizer(new_input_prompt,
                                              return_tensors='pt',
                                              #    max_length=self.text_max_len,
                                              padding='longest',
                                              truncation=False,
                                              return_attention_mask=True)

        is_mol_token = smiles_prompt_tokens.input_ids == self.mol_token_id
        smiles_prompt_tokens['is_mol_token'] = is_mol_token

        

        topk_scores_numpy1 = torch.tensor(np.array(top_k_scores1)) # [batchsize,topk]
        topk_scores_numpy2 = torch.tensor(np.array(top_k_scores2))

        outputss = []
        for item in outputs:
            outputss.append(item[0])


        return graphs1, graphs2, smiles_prompt_tokens, drug1_name, drug2_name, None,topk_scores_numpy1,topk_scores_numpy2,  None, outputss

    def tokenizer_text(self, text):
        sentence_token = self.tokenizer(text=text,
                                        truncation=True,
                                        padding='max_length',
                                        add_special_tokens=True,
                                        max_length=self.text_max_len,
                                        return_tensors='pt',
                                        return_attention_mask=True)
        return sentence_token














#CUSTOM_SEQ_RE = re.compile(r"(\[START_(DNA|SMILES|I_SMILES|MOL|AMINO|FUN)])(.*?)(\[END_\2])")
CUSTOM_SEQ_RE = re.compile(r"(\[START_(DNA|MOL|AMINO|FUN)])(.*?)(\[END_\2])")
SPLIT_MARKER = f"SPL{1}T-TH{1}S-Pl3A5E"


def smiles_handler(text, mol_ph):
    # 在smiles后面加入分子图token占位符。
    smiles_list = []
    for match in CUSTOM_SEQ_RE.finditer(text):
        smiles = match.group(3)
        smiles_list.append(smiles)

    # text = CUSTOM_SEQ_RE.sub(r'\1\3\4%s' % (mol_ph), text)
    text = CUSTOM_SEQ_RE.sub(r'\1\3%s\4' % (mol_ph), text)
    # print("aaaaaaaaaaaa",text)
    # assert test==0

    # aaaaaaaaaaaaaaaa </s> The SMILES of this molecule is
    """
    '<s> The mol of first drug is [START_MOL]<mol><mol><mol><mol><mol><mol><mol><mol>[END_MOL] </s> <s>The mol of second drug is [START_MOL]<mol><mol><mol><mol><mol><mol><mol><mol>[END_MOL] </s>.The first Mechanism Factor is [START_FUN][END_FUN], and the second Mechanism Factor is [START_FUN][END_FUN]. Therefore, the effect of these two drugs is?'
    """

    # text = escape_custom_split_sequence(text)
    return text, smiles_list


def count_subdirectories(folder_path):
    try:
        entries = os.listdir(folder_path)
        subdirectories = [entry for entry in entries if os.path.isdir(os.path.join(folder_path, entry))]

        return len(subdirectories)
    except FileNotFoundError:
        print(f"file {folder_path} not exsit")
        return -1  
    except Exception as e:
        print(f"error:{e}")
        return -2  


def _insert_split_marker(m: re.Match):
    """
    Applies split marker based on a regex match of special tokens such as
    [START_DNA].

    Parameters
    ----------
    n : str
        Input text to split

    Returns
    ----------
    str - the text with the split token added
    """
    start_token, _, sequence, end_token = m.groups()  # the sequence is smiles
    sequence = re.sub(r"(.)", fr"{SPLIT_MARKER}\1", sequence, flags=re.DOTALL)
    return f"{start_token}{sequence}{SPLIT_MARKER}{end_token}"


def escape_custom_split_sequence(text):
    """
    Applies custom splitting to the text for GALILEO's tokenization

    Parameters
    ----------
    text : str
        Input text to split

    Returns
    ----------
    str - the text with the split token added
    """
    return CUSTOM_SEQ_RE.sub(_insert_split_marker, text)


