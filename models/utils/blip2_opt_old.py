"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging
import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType, PeftModel
from ogb.utils import smiles2graph
from torch_geometric.loader.dataloader import Collater
from torch_geometric.data import Data
import numpy as np
from lavis.models.blip2_models.blip2 import (
    # Blip2Base,
    disabled_train,
)
from .blip2 import Blip2Base
from transformers import AutoTokenizer,AutoModel
from transformers import OPTForCausalLM
from torch_geometric.utils import to_dense_batch
# from opendelta import LoraModel
# from opendelta.delta_models.lora import LoraConfig
# from opendelta.delta_configs
torch.set_printoptions(threshold=np.inf)


import re
SPLIT_MARKER = f"SPL{1}T-TH{1}S-Pl3A5E"

CUSTOM_SEQ_RE = re.compile(r"(\[START_(DNA|SMILES|I_SMILES|AMINO)])(.*?)(\[END_\2])")

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
                    #self.model.append(nn.BatchNorm1d(hidden_dims[i + 1]))
                    self.model.append(nn.LayerNorm(hidden_dims[i + 1]))

    def forward(self, h):
        return self.model(h)

class Mol_Adapter(nn.Module):
    def __init__(self,hidden_dim, output_dim):
        super(Mol_Adapter, self).__init__()
        self.W_1 = nn.Linear(hidden_dim, output_dim)
        self.W_2 = nn.Linear(hidden_dim, output_dim)

        self.W_Q = nn.Linear(output_dim, output_dim)
        self.W_K = nn.Linear(output_dim, output_dim)
        self.W_V = nn.Linear(output_dim, output_dim)


    def forward(self, drug1, drug2):
        # adapter for every drug and self-attention
        drug1 = self.W_1(drug1)
        drug2 = self.W_2(drug2)
        drug1_node_num = drug1.shape[1]
        drug_concat = torch.cat((drug1,drug2),1) #[batch,xx,300]
        Q = self.W_Q(drug_concat)
        K = self.W_K(drug_concat)
        V = self.W_V(drug_concat)
        A = Q @ K.transpose(-1, -2) / (Q.size(-1) ** 0.5)
        A = A.softmax(dim=-1)
        out = Q + A @ V
        drug1_new = out[:,:drug1_node_num,:]
        drug2_new = out[:,drug1_node_num:,:]
        #print("drug1_new",drug1_new.shape)
        assert drug1_new.shape[1] == drug1.shape[1]
        assert drug2_new.shape[1] == drug2.shape[1]
        return drug1_new,drug2_new

class Alignment_Mol_Encoder(nn.Module):
    def __init__(self, hidden_dim, num_clusters, residual=False):
        super().__init__()

        self.Q = nn.Parameter(torch.Tensor(1, num_clusters, hidden_dim))
        nn.init.xavier_uniform_(self.Q)

        self.W_Q = nn.Linear(hidden_dim, hidden_dim)
        self.W_K = nn.Linear(hidden_dim, hidden_dim)
        self.W_V = nn.Linear(hidden_dim, hidden_dim)
        self.W_O = nn.Linear(hidden_dim, hidden_dim)


        self.residual = residual
        #self.adapter = MLP( [], hidden_dim, hidden_dim, dropout=0.3)


    def forward(self, x, mask):
        K = self.W_K(x)
        V = self.W_V(x)
        #K, mask = to_dense_batch(K, batch)
        # mask: (batch_size, max_num_nodes)
        #V, _ = to_dense_batch(V, batch)

        attn_mask = (~mask).float().unsqueeze(1)
        attn_mask = attn_mask * (-1e9)


        Q = self.Q.tile(K.size(0), 1, 1)
        Q = self.W_Q(Q)

        A = Q @ K.transpose(-1, -2) / (Q.size(-1) ** 0.5)
        A = A + attn_mask
        A = A.softmax(dim=-1)
        # (batch_size, num_clusters, max_num_nodes)

        out = Q + A @ V

        if self.residual:
            out = out + self.W_O(out).relu()
        else:
            out = self.W_O(out).relu()
        out = F.normalize(out, dim=-1)


        return out

class Function_Adapter(nn.Module):
    def __init__(self,  num_query_token, hidden_size=2048, residual=False):
        super().__init__()
        self.query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, hidden_size)
        )
        self.query_tokens.data.normal_(mean=0.0, std=0.02)
        self.residual = residual
        self.W_Q = nn.Linear(hidden_size, hidden_size)
        self.W_K = nn.Linear(hidden_size, hidden_size)
        self.W_V = nn.Linear(hidden_size, hidden_size)
        self.W_O = nn.Linear(hidden_size, hidden_size)



    def forward(self, q, x):
        # x [batch, token ,dim], q [batch, q_num, dim]
        K = self.W_K(x)
        V = self.W_V(x)

        Q = self.W_Q(q)
        A = Q @ K.transpose(-1, -2) / (Q.size(-1) ** 0.5)
        A = A.softmax(dim=-1)
        out = Q + A @ V

        if self.residual:
            out = out + self.W_O(out).relu()
        else:
            out = self.W_O(out).relu()
        return out

class NtXent(nn.modules.loss._Loss):
    def __init__(self,temperature, return_logits=False):
        super(NtXent, self).__init__()
        self.temperature = temperature
        self.INF = 1e8
        self.return_logits = return_logits

    def forward(self, z_i, z_j):
        #z_i=z_i.cpu().detach()
        #z_j=z_j.cpu().detach()
        N = len(z_i)
        z_i = torch.mean(z_i, dim=1)
        z_j = torch.mean(z_j, dim=1)
        z_i = F.normalize(z_i, p=2, dim=-1) # dim [N,24, D]
        z_j = F.normalize(z_j, p=2, dim=-1) # dim [N,24, D]
        sim_zii= (z_i @ z_i.transpose(-1, -2) ) / self.temperature # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zjj = (z_j @ z_j.transpose(-1, -2)) / self.temperature # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zij = (z_i @ z_j.transpose(-1, -2)) / self.temperature # dim [N, N] => the diag contains the correct pairs (i,j) (x transforms via T_i and T_j)
        # 'Remove' the diag terms by penalizing it (exp(-inf) = 0)
        sim_zii = sim_zii - self.INF * torch.eye(N, device=z_i.device)
        sim_zjj = sim_zjj - self.INF * torch.eye(N, device=z_i.device)
        correct_pairs = torch.arange(N, device=z_i.device).long()
        loss_i = F.cross_entropy(torch.cat([sim_zij, sim_zii], dim=1), correct_pairs)
        loss_j = F.cross_entropy(torch.cat([sim_zij.T, sim_zjj], dim=1), correct_pairs)

        if self.return_logits:
            return (loss_i + loss_j), sim_zij, correct_pairs
        #print((loss_i + loss_j))
        return loss_i + loss_j
    


class Blip2OPT(Blip2Base):
    """
    BLIP2 first-stage model with Q-former and ViT.
    Supported model types:
        - pretrained: pretrained model with vit-g
        - pretrain_vitL: pretrained model with vit-large
        - coco: fintuned model on coco
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2", "pretrain")
    """
    def __init__(
        self,
        bert_name,
        gin_num_layers,
        gin_hidden_dim,
        gin_drop_ratio,
        tune_gnn=False,
        num_query_token=32,
        cross_attention_freq=2,
        llm_tune='freeze',
        peft_dir='',
        opt_model="facebook/galactica-1.3b",
        prompt="",
        args=None,
    ):
        super().__init__()
        self.args = args
        self.use_alignment = args["use_alignment"]

        self.graph_encoder, self.ln_graph = self.init_graph_encoder(gin_num_layers, gin_hidden_dim, gin_drop_ratio)
        self.tune_gnn = tune_gnn
        if not tune_gnn:
            for name, param in self.graph_encoder.named_parameters():
                param.requires_grad = False
            self.graph_encoder = self.graph_encoder.eval()
            self.graph_encoder.train = disabled_train
            logging.info("freeze graph encoder")
        
        self.num_query_token = num_query_token
        self.Qformer, self.query_tokens = self.init_Qformer(bert_name, num_query_token,
                                                            self.graph_encoder.num_features, cross_attention_freq)
        ### remove the unused parameters
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        #####################################
        ## initialize opt model

        opt_model = "./all_checkpoints/galactica-1.3b"
        self.opt_tokenizer = AutoTokenizer.from_pretrained(opt_model, use_fast=False, padding_side='right')
        self.opt_tokenizer.add_special_tokens({'pad_token': '<pad>', 'sep_token': '</s>'})
        self.opt_tokenizer.add_tokens('<mol>') # molecule placeholder
        self.opt_tokenizer.add_tokens('[START_MOL]') # molecule placeholder
        self.opt_tokenizer.add_tokens('[END_MOL]') # molecule placeholder
        self.opt_tokenizer.add_tokens('[START_CoT]') # molecule placeholder
        self.opt_tokenizer.add_tokens('[END_CoT]') # molecule placeholder
        self.opt_tokenizer.add_tokens('<fun>')  # molecule placeholder
        self.opt_tokenizer.add_tokens('[START_FUN]')  # molecule placeholder
        self.opt_tokenizer.add_tokens('[END_FUN]')  # molecule placeholder
        
        #print("[MOL_START] id",self.opt_tokenizer('[MOL_START]').input_id)
        self.mol_token = '<mol>'
        self.fun_token = '<fun>'
        self.opt_tokenizer.mol_token_id = self.opt_tokenizer("<mol>", add_special_tokens=False).input_ids[0]
        self.opt_tokenizer.fun_token_id = self.opt_tokenizer("<fun>", add_special_tokens=False).input_ids[0]
        #print("self.opt_tokenizer.mol_token_id",self.opt_tokenizer.mol_token_id,self.opt_tokenizer.fun_token_id)


        self.collater = Collater([], [])
        
        if opt_model == 'facebook/galactica-125m':
            self.opt_model = OPTForCausalLM.from_pretrained(opt_model)
        else:
            # self.opt_model = OPTForCausalLM.from_pretrained(opt_model, torch_dtype=torch.float16)
            self.opt_model = OPTForCausalLM.from_pretrained(opt_model, torch_dtype=torch.bfloat16)
        self.opt_model.resize_token_embeddings(len(self.opt_tokenizer)) ## this will cause bug when full fine-tuning the opt model

        self.llm_tune = llm_tune
        peft_config_file = False
        if llm_tune == 'lora':
            if peft_dir:
                self.opt_model = PeftModel.from_pretrained(self.opt_model, peft_dir, is_trainable=True)
                #for name, param in self.opt_model.named_parameters():
                #    param.requires_grad = False
            else:
                if peft_config_file:
                    peft_config = LoraConfig(**LoraConfig.from_json_file(peft_config_file))
                else:
                    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False,
                                             r=args["lora_r"], lora_alpha=args["lora_alpha"], lora_dropout=args["lora_dropout"])
                self.peft_config = peft_config
                self.opt_model = get_peft_model(self.opt_model, peft_config)
            self.opt_model.print_trainable_parameters()
        elif llm_tune == 'freeze':
            for name, param in self.opt_model.named_parameters():
                param.requires_grad = False
        elif llm_tune == 'full':
            pass
        else:
            raise NotImplementedError()

        ## fixme: this is different from the original BLIP2
        self.eos_token_id = self.opt_tokenizer(
            "\n", add_special_tokens=False
        ).input_ids[0]

        self.opt_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.opt_model.config.hidden_size
        )
        self.prompt = prompt

        ## function encode
        self.function_encode = Function_Adapter(num_query_token=self.args["function_query_num"], hidden_size=2048, residual=False)
        self.function_query = self.function_encode.query_tokens

        ## use alignment
        if self.use_alignment:
            self.Mol_Adapter = Mol_Adapter(hidden_dim=300,output_dim=2048)
            self.Alignment_Mol_Encoder = Alignment_Mol_Encoder(hidden_dim=300, num_clusters = self.args["function_query_num"], residual=False)
            self.loss_align = NtXent(temperature=0.1)
            #self.loss_align1 = torch.nn.MSELoss(reduction='mean')
            #self.loss_align2 = torch.nn.MSELoss(reduction='mean')


        # self.use_function_generator = args["use_function_generator"]
        # if self.use_function_generator:
        #     self.adapter = MLP([300,512], 300, 768, 0.1)


    def forward(self, batch):

        graphs1, graphs2, prompt_tokens, text_tokens, function1_tokens, function2_tokens = batch

        # encode mol
        graph_embeds1, graph_masks1 = self.graph_encoder(graphs1) ##[batch, 406, 300], [256, 406]
        if not self.tune_gnn:
            graph_embeds1 = graph_embeds1.detach()
        graph_for_adapt1 = graph_embeds1 #[batch,35,300]
        graph_embeds1 = self.ln_graph(graph_embeds1, graph_masks1) ##[batch,35,300]
        query_tokens1 = self.query_tokens.expand(graph_embeds1.shape[0], -1, -1)
        query_output1 = self.Qformer.bert(
            query_embeds=query_tokens1,
            encoder_hidden_states=graph_embeds1,
            encoder_attention_mask=graph_masks1, # fixme: check whether this mask is correct
            return_dict=True)
        mol_tokens1 = self.opt_proj(query_output1.last_hidden_state)

        graph_embeds2, graph_masks2 = self.graph_encoder(graphs2)
        if not self.tune_gnn:
            graph_embeds2 = graph_embeds2.detach()
        graph_for_adapt2 = graph_embeds2
        graph_embeds2 = self.ln_graph(graph_embeds2, graph_masks2)
        device = graph_embeds2.device
        query_tokens2 = self.query_tokens.expand(graph_embeds2.shape[0], -1, -1)
        query_output2 = self.Qformer.bert(
            query_embeds=query_tokens2,
            encoder_hidden_states=graph_embeds2,
            encoder_attention_mask=graph_masks2, # fixme: check whether this mask is correct
            return_dict=True)
        mol_tokens2 = self.opt_proj(query_output2.last_hidden_state)

        mol_tokens = torch.cat([mol_tokens1,mol_tokens2], dim=1)
        mol_tokens = mol_tokens.to(torch.bfloat16)


        # encode function
        function1_embeds = self.opt_model.get_input_embeddings()(function1_tokens.input_ids)
        function2_embeds = self.opt_model.get_input_embeddings()(function2_tokens.input_ids)
        #print("function2_embeds", function2_embeds.shape)
        function_query1 = self.function_query.expand(graph_embeds1.shape[0], -1, -1)
        function_query2 = self.function_query.expand(graph_embeds1.shape[0], -1, -1)
        function_query1 = self.function_encode(function_query1, function1_embeds) # [batch,12,2048]
        function_query2 = self.function_encode(function_query2, function2_embeds)# [batch,12,2048]
        function_tokens = torch.cat([function_query1, function_query2], dim=1) # [batch,24,2048]
        function_tokens = function_tokens.to(torch.bfloat16)


        if self.use_alignment:
            
            graph_for_adapt1 = self.Alignment_Mol_Encoder(graph_for_adapt1, graph_masks1)#[batch,12,2048]
            graph_for_adapt2 = self.Alignment_Mol_Encoder(graph_for_adapt2, graph_masks2)#[batch,12,2048]
            
            drug1_new, drug2_new = self.Mol_Adapter(graph_for_adapt1, graph_for_adapt2 )
            
            #graph_concat_tokens = torch.cat([drug1_new, drug2_new], dim=1)  # [batch,24,2048]
            
            #loss_a1 = self.loss_align1(drug1_new,function_query1)
            #loss_a2 = self.loss_align2(drug2_new,function_query2)
            
            graph_concat_tokens = torch.cat([drug1_new, drug2_new], dim=1) # [batch,24,2048]
            graph_concat_tokens = graph_concat_tokens.to(torch.bfloat16)
            #loss_a = loss_a1+loss_a2

            loss_a = self.loss_align(graph_concat_tokens,function_tokens)
            function_tokens = graph_concat_tokens


        """
        ]function1_tokens torch.Size([4, 29, 2048])
        function2_embeds torch.Size([4, 29, 2048])
        function1_tokens torch.Size([4, 17, 2048])
        function2_embeds torch.Size([4, 9, 2048])

        """


        empty_targets = torch.ones(prompt_tokens.attention_mask.shape, dtype=torch.long).to(device).fill_(-100)
        targets = text_tokens.input_ids.masked_fill(
            text_tokens.input_ids == self.opt_tokenizer.pad_token_id, -100
        )
        targets = torch.cat([empty_targets, targets], dim=1)
        prompt_embeds = self.opt_model.get_input_embeddings()(prompt_tokens.input_ids)


        prompt_embeds[prompt_tokens.is_mol_token] = mol_tokens.flatten(0, 1)
        prompt_embeds[prompt_tokens.is_fun_token] = function_tokens.flatten(0, 1) ## prompt_tokens.is_fun_token[batch,95],function_tokens.flatten(0, 1).shape#[96,2048]

        inputs_embeds = self.opt_model.get_input_embeddings()(text_tokens.input_ids)
        #print("inputs_embeds",inputs_embeds.shape)
        inputs_embeds = torch.cat((prompt_embeds, inputs_embeds), dim=1)
        #print("inputs_embeds",inputs_embeds.shape)
        attention_mask = torch.cat([prompt_tokens.attention_mask, text_tokens.attention_mask], dim=1)

        """
        function1_tokens torch.Size([4, 9])
        function1_tokens torch.Size([4, 9, 2048])
        prompt_embeds torch.Size([4, 93, 2048])
        mol_tokens torch.Size([4, 16, 2048])
        inputs_embeds torch.Size([4, 18, 2048])
        inputs_embeds torch.Size([4, 111, 2048])
        function1_tokens torch.Size([4, 22])
        function1_tokens torch.Size([4, 22, 2048])
        prompt_embeds torch.Size([4, 93, 2048])
        mol_tokens torch.Size([4, 16, 2048])s
        inputs_embeds torch.Size([4, 19, 2048])
        inputs_embeds torch.Size([4, 112, 2048])

        """

        outputs = self.opt_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,  #改成functiopn的token
        )
        loss = outputs.loss#分子-function loss1 权重0.5
        logit=outputs.logit1
        outputs1 = self.opt_model(
            inputs_embeds=(inputs_embeds,logit),
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )
        losszong=outputs1.loss
        #print("output",outputs.logits.shape)
        loss = outputs.loss
        if self.use_alignment:
            loss = loss_a+loss
        else:
            loss = loss

        #print("loss",loss)
        return {"loss": loss}
        # else:
        #     graphs1, graphs2, prompt_tokens, text_tokens, function_emb = batch
        #     graph_embeds1, graph_masks1 = self.graph_encoder(graphs1)  ##[256, 406, 300],[256, 406]
        #     if not self.tune_gnn:
        #         graph_embeds1 = graph_embeds1.detach()
        #     graph_embeds1 = self.ln_graph(graph_embeds1, graph_masks1)
        #     # get drug1 function
        #     print("dddddddddd",len(function_emb))
        #     self.function_adapter(graph_embeds1, function_emb.to(graph_embeds1.device))


    @torch.no_grad()
    def generate(
        self,
        samples,
        do_sample=False,
        num_beams=5,
        max_length=128,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        
        graphs1 = samples['graphs1']
        prompt_tokens = samples['prompt_tokens']
        # prompt_lens = samples['prompt_lens']
        # with self.maybe_autocast():
        graph_embeds1, graph_masks1 = self.graph_encoder(graphs1)
        graph_for_adapt1 = graph_embeds1 #[batch,35,300]

        graph_embeds1 = self.ln_graph(graph_embeds1)

        query_tokens1 = self.query_tokens.expand(graph_embeds1.shape[0], -1, -1)
        query_output1 = self.Qformer.bert(
            query_embeds=query_tokens1,
            encoder_hidden_states=graph_embeds1,
            encoder_attention_mask=graph_masks1,
            return_dict=True,
        )
        mol_tokens1 = self.opt_proj(query_output1.last_hidden_state)

        graphs2 = samples['graphs2']
        # prompt_lens = samples['prompt_lens']
        # with self.maybe_autocast():
        graph_embeds2, graph_masks2 = self.graph_encoder(graphs2)
        graph_for_adapt2 = graph_embeds2
        graph_embeds2 = self.ln_graph(graph_embeds2)

        query_tokens2 = self.query_tokens.expand(graph_embeds2.shape[0], -1, -1)
        query_output2 = self.Qformer.bert(
            query_embeds=query_tokens2,
            encoder_hidden_states=graph_embeds2,
            encoder_attention_mask=graph_masks2,
            return_dict=True,
        )
        mol_tokens2 = self.opt_proj(query_output2.last_hidden_state)
        
        mol_tokens=torch.cat([mol_tokens1,mol_tokens2],dim=1)
        prompt_embeds = self.opt_model.get_input_embeddings()(prompt_tokens.input_ids)
        prompt_embeds[prompt_tokens.is_mol_token] = mol_tokens.flatten(0, 1)




        # encode function
        if self.use_alignment:
            
            # drug1_new, drug2_new = self.Mol_Adapter(graph_for_adapt1,graph_for_adapt2 )
            # graph_for_adapt1 = self.Alignment_Mol_Encoder(drug1_new, graph_masks1)#[batch,12,300]
            # graph_for_adapt2 = self.Alignment_Mol_Encoder(drug2_new, graph_masks2)#[batch,12,300]
            # graph_concat_tokens = torch.cat([graph_for_adapt1, graph_for_adapt2], dim=1)  # [batch,24,300]
            # function_tokens = graph_concat_tokens
            # function_tokens = function_tokens.to(torch.bfloat16)
            
            graph_for_adapt1 = self.Alignment_Mol_Encoder(graph_for_adapt1, graph_masks1)#[batch,12,2048]
            graph_for_adapt2 = self.Alignment_Mol_Encoder(graph_for_adapt2, graph_masks2)#[batch,12,2048]
            
            drug1_new, drug2_new = self.Mol_Adapter(graph_for_adapt1,graph_for_adapt2 )
            
            #graph_concat_tokens = torch.cat([graph_for_adapt1, graph_for_adapt2], dim=1)  # [batch,24,2048]
            #loss_a1 = self.loss_align1(drug1_new,function_query1)
            #loss_a2 = self.loss_align2(drug2_new,function_query2)
            
            function_tokens = torch.cat([drug1_new, drug2_new], dim=1) # [batch,24,2048]
            function_tokens = function_tokens.to(torch.bfloat16)
            
            
        else:
            function1_tokens = samples['function_token1']
            function2_tokens = samples['function_token2']
            function1_embeds = self.opt_model.get_input_embeddings()(function1_tokens.input_ids)
            function2_embeds = self.opt_model.get_input_embeddings()(function2_tokens.input_ids)
            #print("function2_embeds", function2_embeds.shape)
            function_query1 = self.function_query.expand(graph_embeds1.shape[0], -1, -1)
            function_query2 = self.function_query.expand(graph_embeds1.shape[0], -1, -1)
            function_query1 = self.function_encode(function_query1, function1_embeds) # [batch,12,2048]
            function_query2 = self.function_encode(function_query2, function2_embeds)# [batch,12,2048]
            function_tokens = torch.cat([function_query1, function_query2], dim=1) # [batch,24,2048]
            function_tokens = function_tokens.to(torch.bfloat16)
        prompt_embeds[prompt_tokens.is_fun_token] = function_tokens.flatten(0, 1) ##



        #print("&&&&&&&&&&&&&&&&&&&&&&&&&&",prompt_embeds)
        #repetition_penalty=1.5
        outputs = self.opt_model.generate(
            inputs_embeds=prompt_embeds,
            attention_mask=prompt_tokens.attention_mask,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            num_beams=num_beams,
            #max_length=max_length,
            max_new_tokens=max_length,
            min_length=min_length,
            # pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_return_sequences=num_captions,
            # use_cache=False,
        )
        output_text = self.opt_tokenizer.batch_decode(outputs, skip_special_tokens=True)

        output_text = [text.strip() for text in output_text]
        return output_text
        

    @torch.no_grad()
    def blip_qa(
        self, 
        samples,
        do_sample=False,
        num_beams=5,
        max_length=128,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
        output_scores=False,
        ):

        device = next(self.parameters()).device
        
        ## data processing
        prompts = samples['prompts'] # assume list of strings
        prepared_prompts = []
        mol_list = []
        for p in prompts:
            text, smiles = smiles_handler(p, self.mol_token * self.num_query_token)
            prepared_prompts.append(text)
            mol_list.extend([smiles2data(s) for s in smiles])
        
        prompt_tokens = self.opt_tokenizer(prepared_prompts,
                                           truncation=False,
                                           padding='longest',
                                           add_special_tokens=True,
                                        #    max_length=self.args.max_len[],
                                           return_tensors='pt',
                                           return_attention_mask=True).to(device)
        
        ## forward function
        prompt_embeds = self.opt_model.get_input_embeddings()(prompt_tokens.input_ids)
        
        if len(mol_list) > 0:
            graphs = self.collater(mol_list).to(device)
            is_mol_token = (prompt_tokens.input_ids == self.mol_token) # shape = [B, max_len]
            ## graph forward
            graph_embeds, graph_masks = self.graph_encoder(graphs)
            graph_embeds = self.ln_graph(graph_embeds, graph_masks)
            query_tokens = self.query_tokens.expand(graph_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=graph_embeds,
                encoder_attention_mask=graph_masks, # fixme: check whether this mask is correct
                return_dict=True,
            )
            mol_tokens = self.opt_proj(query_output.last_hidden_state) # shape = [mol_num, num_query_token, D]
            ## replace mol tokens
            prompt_embeds[is_mol_token] = mol_tokens.flatten(0, 1)
        
        if output_scores:
            outputs = self.opt_model.generate(
                    inputs_embeds=prompt_embeds,
                    attention_mask=prompt_tokens.attention_mask,
                    do_sample=do_sample,
                    top_p=top_p,
                    temperature=temperature,
                    num_beams=num_beams,
                    max_length=max_length,
                    min_length=min_length,
                    # pad_token_id=self.pad_token_id,
                    eos_token_id=self.eos_token_id,
                    repetition_penalty=repetition_penalty,
                    length_penalty=length_penalty,
                    num_return_sequences=num_captions,
                    output_scores=True,
                    return_dict_in_generate=True
                    # use_cache=False,
            )
            return outputs
        else:
            outputs = self.opt_model.generate(
                    inputs_embeds=prompt_embeds,
                    attention_mask=prompt_tokens.attention_mask,
                    do_sample=do_sample,
                    top_p=top_p,
                    temperature=temperature,
                    num_beams=num_beams,
                    max_length=max_length,
                    min_length=min_length,
                    # pad_token_id=self.pad_token_id,
                    eos_token_id=self.eos_token_id,
                    repetition_penalty=repetition_penalty,
                    length_penalty=length_penalty,
                    num_return_sequences=num_captions,
                    # use_cache=False,
                )
            output_text = self.opt_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            output_text = [text.strip() for text in output_text]
            return output_text
    
    @torch.no_grad()
    def opt_qa(
        self, 
        samples,
        do_sample=False,
        num_beams=5,
        max_length=128,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
        output_scores=False,
        ):

        device = next(self.parameters()).device
        ## data processing
        prompts = samples['prompts'] # assume list of strings
        prompts = [escape_custom_split_sequence(p) for p in prompts]
        
        prompt_tokens = self.opt_tokenizer(prompts,
                                           truncation=False,
                                           padding='longest',
                                           add_special_tokens=True,
                                        #    max_length=self.args.max_len[],
                                           return_tensors='pt',
                                           return_attention_mask=True).to(device)
        
        prompt_embeds = self.opt_model.get_input_embeddings()(prompt_tokens.input_ids)

        if output_scores:
            ## forward function
            outputs = self.opt_model.generate(
                    inputs_embeds=prompt_embeds,
                    attention_mask=prompt_tokens.attention_mask,
                    do_sample=do_sample,
                    top_p=top_p,
                    temperature=temperature,
                    num_beams=num_beams,
                    max_length=max_length,
                    min_length=min_length,
                    # pad_token_id=self.pad_token_id,
                    eos_token_id=self.eos_token_id,
                    repetition_penalty=repetition_penalty,
                    length_penalty=length_penalty,
                    num_return_sequences=num_captions,
                    # use_cache=False,
                    output_scores=True,
                    return_dict_in_generate=True
                )
            return outputs
        else:
            ## forward function
            outputs = self.opt_model.generate(
                    inputs_embeds=prompt_embeds,
                    attention_mask=prompt_tokens.attention_mask,
                    do_sample=do_sample,
                    top_p=top_p,
                    temperature=temperature,
                    num_beams=num_beams,
                    max_length=max_length,
                    min_length=min_length,
                    # pad_token_id=self.pad_token_id,
                    eos_token_id=self.eos_token_id,
                    repetition_penalty=repetition_penalty,
                    length_penalty=length_penalty,
                    num_return_sequences=num_captions,
                    # use_cache=False,
                )
            output_text = self.opt_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            output_text = [text.strip() for text in output_text]
            return output_text
        
    @torch.no_grad()
    def probe_qformer(
        self, 
        batch,
        do_sample=False,
        num_beams=5,
        max_length=128,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
        ):
        with self.maybe_autocast():
            device = next(self.parameters()).device
            
            graphs, smiles_prompt_tokens, texts = batch
            graphs = graphs.to(device)
            ## graph forward
            graph_embeds, graph_masks = self.graph_encoder(graphs)
            graph_embeds = self.ln_graph(graph_embeds, graph_masks)
            query_tokens = self.query_tokens.expand(graph_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=graph_embeds,
                encoder_attention_mask=graph_masks, # fixme: check whether this mask is correct
                return_dict=True,
            )
            mol_tokens = self.opt_proj(query_output.last_hidden_state) # shape = [mol_num, num_query_token, D]
            B, num_q, D = mol_tokens.shape
            
            ## 
            embed_func = self.opt_model.get_input_embeddings()
            embed_weight = embed_func.weight # shape = [vocab_size, D]
            
            dis_metric = 'cos'
            topk = 10
            if dis_metric == 'cos':
                mol_tokens = F.normalize(mol_tokens, dim=-1, p=2)
                embed_weight = F.normalize(embed_weight, dim=-1, p=2)
                sim = mol_tokens.flatten(0, 1) @ embed_weight.T # shape = [mol_num * num_query_token, vocab_size]
            elif dis_metric == 'euc':
                sim = - torch.cdist(mol_tokens.flatten(0, 1), embed_weight, p=2)
                assert sim.shape == (B * num_q, embed_weight.shape[0])
            else:
                raise NotImplementedError()
            _, topk_ids = torch.topk(sim, k=topk, dim=-1) # shape = [mol_num * num_query_token, k]
            knn_decode_strings = self.opt_tokenizer.batch_decode(topk_ids.flatten())
            knn_decode_strings = np.asarray(knn_decode_strings).reshape(B, num_q, topk).tolist() # shape = [mol_num, num_query_token, topk]
            knn_decode_strings = [[' '.join(ii) for ii in i] for i in knn_decode_strings] # shape = [mol_num, num_query_token]
            if False:
                ### print for presentation
                assert len(knn_decode_strings) == len(texts)
                for predict, text in zip(knn_decode_strings, texts):
                    print('----------------------------')
                    print(predict)
                    print(text)
            return knn_decode_strings


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
    start_token, _, sequence, end_token = m.groups()
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


def smiles_handler(text, mol_ph):
    smiles_list = []
    for match in CUSTOM_SEQ_RE.finditer(text):
        smiles = match.group(3)
        smiles_list.append(smiles)

    text = CUSTOM_SEQ_RE.sub(r'\1\3\4%s' % (mol_ph), text)
    text = escape_custom_split_sequence(text)
    return text, smiles_list



def mask_by_len(input, lens, fill_value=0):
    '''
    input: shape = [N, D]
    lens: shape = [N]
    '''
    mask = torch.arange(input.shape[1], device=input.device).reshape(1, -1)
    mask = mask < lens.reshape(-1, 1)
    input[mask] = fill_value
    return input


def smiles2data(smiles):
    graph = smiles2graph(smiles)
    x = torch.from_numpy(graph['node_feat'])
    edge_index = torch.from_numpy(graph['edge_index'], )
    edge_attr = torch.from_numpy(graph['edge_feat'])
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data
