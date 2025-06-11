
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
from .utils.blip2 import Blip2Base
from transformers import AutoTokenizer, AutoModel
from transformers import OPTForCausalLM
from torch_geometric.utils import to_dense_batch
from typing import List, Optional, Tuple, Union
from easydict import EasyDict
# from opendelta import LoraModel
# from opendelta.delta_models.lora import LoraConfig
# from opendelta.delta_configs
torch.set_printoptions(threshold=np.inf)
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils.generic import ModelOutput
from dataclasses import dataclass
import re
import pickle
import copy

SPLIT_MARKER = f"SPL{1}T-TH{1}S-Pl3A5E"

CUSTOM_SEQ_RE = re.compile(r"(\[START_(DNA|SMILES|I_SMILES|AMINO)])(.*?)(\[END_\2])")




class Blip2OPT_RETRIEVAL(Blip2Base):
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

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.stage = cfg["stage"]
        self.tune_gnn = cfg["tune_gnn"]
        self.llm_tune = cfg["llm_tune"]
        self.bert_name = cfg["bert_name"]
        self.gin_num_layers = cfg["gin_num_layers"]
        self.gin_hidden_dim = cfg["gin_hidden_dim"]
        self.gin_drop_ratio = cfg["gin_drop_ratio"]
        self.num_query_token = cfg["num_query_token"]  # smiles emb token
        self.cross_attention_freq = cfg['cross_attention_freq']
        self.peft_dir = cfg["peft_dir"]
        self.top_k = cfg["retrieval_function_number"]
        print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaself.top_k",self.top_k)

        ### init graph encoder
        self.graph_encoder, self.ln_graph = self.init_graph_encoder(self.gin_num_layers, self.gin_hidden_dim,
                                                                    self.gin_drop_ratio)
        if not self.tune_gnn:
            for name, param in self.graph_encoder.named_parameters():
                param.requires_grad = False
            self.graph_encoder = self.graph_encoder.eval()
            self.graph_encoder.train = disabled_train
            logging.info("freeze graph encoder")

        ### init Q-former
        self.Qformer, self.query_tokens = self.init_Qformer(self.bert_name, self.num_query_token,
                                                            self.graph_encoder.num_features, self.cross_attention_freq)
        ### remove the unused parameters
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        if self.stage == "first":
            for name, param in self.Qformer.named_parameters():
                param.requires_grad = False
            self.Qformer = self.Qformer.eval()
            # self.graph_encoder.train = disabled_train
            logging.info("freeze Qformer encoder")

        #####################################
        ## initialize opt model
        opt_model = "./all_checkpoints/galactica-1.3b"
        self.opt_tokenizer = AutoTokenizer.from_pretrained(opt_model, use_fast=False, padding_side='right')
        self.opt_tokenizer.add_special_tokens({'pad_token': '<pad>', 'sep_token': '</s>'})
        self.opt_tokenizer.add_special_tokens({'eos_token': "<EOS>"})
        self.eos_token_id = self.opt_tokenizer("<EOS>", add_special_tokens=True).input_ids[0]
        

        self.opt_tokenizer.add_special_tokens(
            {'additional_special_tokens': ['<p>', '</p>', '[START_CoT]', '[END_CoT]', '[START_MOL]', '[END_MOL]']})
        self.opt_tokenizer.add_tokens('<mol>')  # molecule placeholder

        self.mol_token = '<mol>'
        self.opt_tokenizer.mol_token_id = self.opt_tokenizer("<mol>", add_special_tokens=False).input_ids[0]  # 50003

        self.collater = Collater([], [])

        if opt_model == 'facebook/galactica-125m':
            # self.opt_model = OPTForCausalLM.from_pretrained(opt_model)
            self.opt_model = OPTForCausalLM_New.from_pretrained(opt_model)
        else:
            # self.opt_model = OPTForCausalLM.from_pretrained(opt_model, torch_dtype=torch.float16)
            # self.opt_model = OPTForCausalLM.from_pretrained(opt_model, torch_dtype=torch.bfloat16)
            self.opt_model = OPTForCausalLM_New.from_pretrained(opt_model, torch_dtype=torch.bfloat16)
        self.opt_model.resize_token_embeddings(
            len(self.opt_tokenizer))  ## this will cause bug when full fine-tuning the opt model

        peft_config_file = False
        if self.llm_tune == 'lora':
            if self.peft_dir:
                self.opt_model = PeftModel.from_pretrained(self.opt_model, self.peft_dir, is_trainable=True)
                # for name, param in self.opt_model.named_parameters():
                #    param.requires_grad = False
            else:
                if peft_config_file:
                    peft_config = LoraConfig(**LoraConfig.from_json_file(peft_config_file))
                else:
                    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False,
                                             r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"],
                                             lora_dropout=cfg["lora_dropout"])
                self.peft_config = peft_config
                self.opt_model = get_peft_model(self.opt_model, peft_config)
            self.opt_model.print_trainable_parameters()
        elif self.llm_tune == 'freeze':
            for name, param in self.opt_model.named_parameters():
                param.requires_grad = False
        elif self.llm_tune == 'full':
            pass
        else:
            raise NotImplementedError()

        self.opt_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.opt_model.config.hidden_size
        )

        ## retrieval

    def forward(self, batch):

        # the inputs
        graphs1, graphs2, smiles_prompt_tokens, drug1_name, drug2_name, output_tokens,topk_scores1, topk_scores2 = batch
        ## Encode Mol
        # mol1
        graph_embeds1, graph_masks1 = self.graph_encoder(graphs1)  ##[batch, 406, 300], [256, 406]
        if not self.tune_gnn:
            graph_embeds1 = graph_embeds1.detach()
        graph_for_adapt1 = graph_embeds1  # [batch,35,300]
        graph_embeds1 = self.ln_graph(graph_embeds1, graph_masks1)  ##[batch,35,300]
        query_tokens1 = self.query_tokens.expand(graph_embeds1.shape[0], -1, -1)
        query_output1 = self.Qformer.bert(
            query_embeds=query_tokens1,
            encoder_hidden_states=graph_embeds1,
            encoder_attention_mask=graph_masks1,  # fixme: check whether this mask is correct
            return_dict=True)
        mol_tokens1 = self.opt_proj(query_output1.last_hidden_state)
        #print("mol_tokens1",mol_tokens1.shape)

        # mol2
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
            encoder_attention_mask=graph_masks2,  # fixme: check whether this mask is correct
            return_dict=True)
        mol_tokens2 = self.opt_proj(query_output2.last_hidden_state)

        mol_tokens = torch.cat([mol_tokens1, mol_tokens2], dim=1)
        mol_tokens = mol_tokens.to(torch.bfloat16)

        # for mol prompts
        mol_prompt_embeds = self.opt_model.get_input_embeddings()(smiles_prompt_tokens.input_ids)  # mol+question1
        mol_prompt_embeds[smiles_prompt_tokens.is_mol_token] = mol_tokens.flatten(0, 1)

        ### for first stage
        input_embeddings11 = torch.cat(
            (mol_prompt_embeds, self.opt_model.get_input_embeddings()(output_tokens.input_ids)), dim=1)

        attention_mask11 = torch.cat([smiles_prompt_tokens.attention_mask, output_tokens.attention_mask], dim=1)

        empty_targets1 = torch.ones(smiles_prompt_tokens.attention_mask.shape, dtype=torch.long).to(device).fill_(-100)
        targets1 = output_tokens.input_ids.masked_fill(
            output_tokens.input_ids == self.opt_tokenizer.pad_token_id, -100
        )
        # bos_token_id = self.opt_model.config.bos_token_id # 0
        # use_bos = bos_token_id is not None and targets1[:, 0].eq(bos_token_id).all() #false
        
        targets1 = torch.cat([empty_targets1, targets1], dim=1)
        outputs1 = self.opt_model(
            inputs_embeds=input_embeddings11,
            attention_mask=attention_mask11,
            return_dict=True,
            labels=targets1,
        )
        
        loss = outputs1.loss 

        return {"loss": loss}

    @torch.no_grad()
    def generate(
            self,
            batch,
            do_sample=False,
            num_beams=5,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.0,
            length_penalty=1.0,
            num_captions=1,
            temperature=1,
            max_new_tokens=25
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
        # load drug1

        
        graphs1, graphs2, mol_prompt_tokens, drug1_name, drug2_name, output_tokens,topk_scores_numpy1,topk_scores_numpy2, gt_CoTs, gt_outputs = batch

        graph_embeds1, graph_masks1 = self.graph_encoder(graphs1) #[batchsize*topk,33, 300]
        graph_embeds1 = self.ln_graph(graph_embeds1)
        query_tokens1 = self.query_tokens.expand(graph_embeds1.shape[0], -1, -1)
        query_output1 = self.Qformer.bert(
            query_embeds=query_tokens1,
            encoder_hidden_states=graph_embeds1,
            encoder_attention_mask=graph_masks1,
            return_dict=True,
        )
        mol_tokens1 = self.opt_proj(query_output1.last_hidden_state)
        # load drug2
        graph_embeds2, graph_masks2 = self.graph_encoder(graphs2)
        graph_embeds2 = self.ln_graph(graph_embeds2)
        query_tokens2 = self.query_tokens.expand(graph_embeds2.shape[0], -1, -1)
        query_output2 = self.Qformer.bert(
            query_embeds=query_tokens2,
            encoder_hidden_states=graph_embeds2,
            encoder_attention_mask=graph_masks2,
            return_dict=True,
        )
        mol_tokens2 = self.opt_proj(query_output2.last_hidden_state)
        # concat mol1 and mol2
        mol_emb = torch.cat([mol_tokens1, mol_tokens2], dim=1)

        # the prompt insert the mol embeddings
        mol_prompt_embeds = self.opt_model.get_input_embeddings()(mol_prompt_tokens.input_ids)
        mol_prompt_embeds[mol_prompt_tokens.is_mol_token] = mol_emb.flatten(0, 1)

        ## stage1
        outputs1 = self.opt_model.generate(
            inputs_embeds=mol_prompt_embeds,
            attention_mask=mol_prompt_tokens.attention_mask,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            min_length=min_length,
            # pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_return_sequences=num_captions,
            #return_dict=True
            #return_dict_in_generate=True,
            #output_scores=True
        )

        outputs1 = self.opt_tokenizer.batch_decode(outputs1, skip_special_tokens=True)
        ddie_text = [text.strip() for text in outputs1]
        # #print("ddie_text",len(ddie_text))      

        return None, ddie_text,drug1_name, drug2_name


        




class Blip2OPT_RETRIEVAL_marginalize(Blip2Base):
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

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.stage = cfg["stage"]
        self.tune_gnn = cfg["tune_gnn"]
        self.llm_tune = cfg["llm_tune"]
        self.bert_name = cfg["bert_name"]
        self.gin_num_layers = cfg["gin_num_layers"]
        self.gin_hidden_dim = cfg["gin_hidden_dim"]
        self.gin_drop_ratio = cfg["gin_drop_ratio"]
        self.num_query_token = cfg["num_query_token"]  # smiles emb token
        self.cross_attention_freq = cfg['cross_attention_freq']
        self.peft_dir = cfg["peft_dir"]
        self.top_k = cfg["retrieval_function_number"]
        print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaself.top_k", self.top_k)

        ### init graph encoder
        self.graph_encoder, self.ln_graph = self.init_graph_encoder(self.gin_num_layers, self.gin_hidden_dim,
                                                                    self.gin_drop_ratio)
        if not self.tune_gnn:
            for name, param in self.graph_encoder.named_parameters():
                param.requires_grad = False
            self.graph_encoder = self.graph_encoder.eval()
            self.graph_encoder.train = disabled_train
            logging.info("freeze graph encoder")

        ### init Q-former
        self.Qformer, self.query_tokens = self.init_Qformer(self.bert_name, self.num_query_token,
                                                            self.graph_encoder.num_features, self.cross_attention_freq)
        ### remove the unused parameters
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        if self.stage == "first":
            for name, param in self.Qformer.named_parameters():
                param.requires_grad = False
            self.Qformer = self.Qformer.eval()
            # self.graph_encoder.train = disabled_train
            logging.info("freeze Qformer encoder")

        #####################################
        ## initialize opt model
        opt_model = "./all_checkpoints/galactica-1.3b"
        self.opt_tokenizer = AutoTokenizer.from_pretrained(opt_model, use_fast=False, padding_side='right')
        self.opt_tokenizer.add_special_tokens({'pad_token': '<pad>', 'sep_token': '</s>'})
        self.opt_tokenizer.add_special_tokens({'eos_token': "<EOS>"})
        self.eos_token_id = self.opt_tokenizer("<EOS>", add_special_tokens=True).input_ids[0]

        self.opt_tokenizer.add_special_tokens(
            {'additional_special_tokens': ['<p>', '</p>', '[START_CoT]', '[END_CoT]', '[START_MOL]', '[END_MOL]']})
        self.opt_tokenizer.add_tokens('<mol>')  # molecule placeholder

        self.mol_token = '<mol>'
        self.opt_tokenizer.mol_token_id = self.opt_tokenizer("<mol>", add_special_tokens=False).input_ids[0]  # 50003

        self.collater = Collater([], [])

        if opt_model == 'facebook/galactica-125m':
            self.opt_model = OPTForCausalLM_New.from_pretrained(opt_model)
        else:
            # self.opt_model = OPTForCausalLM.from_pretrained(opt_model, torch_dtype=torch.float16)
            # self.opt_model = OPTForCausalLM.from_pretrained(opt_model, torch_dtype=torch.bfloat16)
            self.opt_model = OPTForCausalLM_New.from_pretrained(opt_model, torch_dtype=torch.bfloat16)
        self.opt_model.resize_token_embeddings(
            len(self.opt_tokenizer))  ## this will cause bug when full fine-tuning the opt model

        peft_config_file = False
        if self.llm_tune == 'lora':
            if self.peft_dir:
                self.opt_model = PeftModel.from_pretrained(self.opt_model, self.peft_dir, is_trainable=True)
                # for name, param in self.opt_model.named_parameters():
                #    param.requires_grad = False
            else:
                if peft_config_file:
                    peft_config = LoraConfig(**LoraConfig.from_json_file(peft_config_file))
                else:
                    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False,
                                             r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"],
                                             lora_dropout=cfg["lora_dropout"])
                self.peft_config = peft_config
                self.opt_model = get_peft_model(self.opt_model, peft_config)
            self.opt_model.print_trainable_parameters()
        elif self.llm_tune == 'freeze':
            for name, param in self.opt_model.named_parameters():
                param.requires_grad = False
        elif self.llm_tune == 'full':
            pass
        else:
            raise NotImplementedError()

        self.opt_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.opt_model.config.hidden_size
        )

        ## retrieval

    def forward(self, batch):

        # the inputs
        graphs1, graphs2, smiles_prompt_tokens, CoT_prompt_tokens, Question2_tokens, output_tokens, topk_scores1, topk_scores2 = batch
        ## Encode Mol
        # mol1
        graph_embeds1, graph_masks1 = self.graph_encoder(graphs1)  
        if not self.tune_gnn:
            graph_embeds1 = graph_embeds1.detach()
        graph_for_adapt1 = graph_embeds1  
        graph_embeds1 = self.ln_graph(graph_embeds1, graph_masks1)  
        query_tokens1 = self.query_tokens.expand(graph_embeds1.shape[0], -1, -1)
        query_output1 = self.Qformer.bert(
            query_embeds=query_tokens1,
            encoder_hidden_states=graph_embeds1,
            encoder_attention_mask=graph_masks1,  # fixme: check whether this mask is correct
            return_dict=True)
        mol_tokens1 = self.opt_proj(query_output1.last_hidden_state)

        # mol2
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
            encoder_attention_mask=graph_masks2,  # fixme: check whether this mask is correct
            return_dict=True)
        mol_tokens2 = self.opt_proj(query_output2.last_hidden_state)

        mol_tokens = torch.cat([mol_tokens1, mol_tokens2], dim=1)
        mol_tokens = mol_tokens.to(torch.bfloat16)


        # for mol prompts
        mol_prompt_embeds = self.opt_model.get_input_embeddings()(smiles_prompt_tokens.input_ids) 
        mol_prompt_embeds[smiles_prompt_tokens.is_mol_token] = mol_tokens.flatten(0, 1)
        ### for first stage
        input_embeddings111 = torch.cat(
            (mol_prompt_embeds, self.opt_model.get_input_embeddings()(output_tokens.input_ids)), dim=1)

        attention_mask111 = torch.cat([smiles_prompt_tokens.attention_mask, output_tokens.attention_mask], dim=1)

        empty_targets1 = torch.ones(smiles_prompt_tokens.attention_mask.shape, dtype=torch.long).to(device).fill_(-100)
        targets1 = output_tokens.input_ids.masked_fill(
            output_tokens.input_ids == self.opt_tokenizer.pad_token_id, -100
        )
       
        targets1 = torch.cat([empty_targets1, targets1], dim=1)
        outputs1 = self.opt_model(
            inputs_embeds=input_embeddings111,
            attention_mask=attention_mask111,
            return_dict=True,
            labels=targets1,
        )

        logits = outputs1.logits  
        doc_logprobs1 = nn.functional.softmax(topk_scores1[:, :self.top_k], dim=1)  
        doc_logprobs2 = nn.functional.softmax(topk_scores2[:, :self.top_k], dim=1)
        full_doc_logprobs = torch.einsum("ab,ac->abc", doc_logprobs1, doc_logprobs2).view(
            doc_logprobs2.shape[0], -1)  # [4,9]

        loss = self.token_loss(
            seq_logits=logits,
            doc_logprobs=full_doc_logprobs,
            target=targets1,
            exclude_bos_score=False,
            n_docs=self.top_k*self.top_k,
            retrieval_labels=targets1,
        )

        return {"loss": loss}

    @torch.no_grad()
    def token_generate(
            self,
            batch,
            do_sample=False,
            num_beams=5,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.0,
            length_penalty=1.0,
            num_captions=1,
            temperature=1,
            max_new_tokens=25
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
        # load drug1

        n_docs = self.top_k * self.top_k

        graphs1, graphs2, mol_prompt_tokens, CoT_prompt_tokens, Question2_tokens, output_tokens, topk_scores_numpy1, topk_scores_numpy2, gt_outputs, gt_CoTs = batch

        graph_embeds1, graph_masks1 = self.graph_encoder(graphs1)
        graph_embeds1 = self.ln_graph(graph_embeds1)
        query_tokens1 = self.query_tokens.expand(graph_embeds1.shape[0], -1, -1)
        query_output1 = self.Qformer.bert(
            query_embeds=query_tokens1,
            encoder_hidden_states=graph_embeds1,
            encoder_attention_mask=graph_masks1,
            return_dict=True,
        )
        mol_tokens1 = self.opt_proj(query_output1.last_hidden_state)
        # load drug2
        graph_embeds2, graph_masks2 = self.graph_encoder(graphs2)
        graph_embeds2 = self.ln_graph(graph_embeds2)
        query_tokens2 = self.query_tokens.expand(graph_embeds2.shape[0], -1, -1)
        query_output2 = self.Qformer.bert(
            query_embeds=query_tokens2,
            encoder_hidden_states=graph_embeds2,
            encoder_attention_mask=graph_masks2,
            return_dict=True,
        )
        mol_tokens2 = self.opt_proj(query_output2.last_hidden_state)
        # concat mol1 and mol2
        mol_emb = torch.cat([mol_tokens1, mol_tokens2], dim=1)

        # the prompt insert the mol embeddings
        mol_prompt_embeds = self.opt_model.get_input_embeddings()(mol_prompt_tokens.input_ids)
        mol_prompt_embeds[mol_prompt_tokens.is_mol_token] = mol_emb.flatten(0, 1)

        input_embeddings1 = torch.cat(
            (mol_prompt_embeds, self.opt_model.get_input_embeddings()(CoT_prompt_tokens.input_ids)), dim=1)

        attention_mask1 = torch.cat([mol_prompt_tokens.attention_mask, CoT_prompt_tokens.attention_mask], dim=1)

        ## stage1
        outputs1 = self.opt_model.generate(
            inputs_embeds=input_embeddings1,
            attention_mask=attention_mask1,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            min_length=min_length,
            # pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_return_sequences=num_captions,
            # use_cache=False,
        )

        ddie_text = self.opt_tokenizer.batch_decode(outputs1, skip_special_tokens=True)
        ddie_text = [text.strip() for text in ddie_text]

        return None, ddie_text

    def token_loss(
            self, seq_logits, doc_logprobs, target, reduce_loss=True, epsilon=0.0, exclude_bos_score=False,
            ignore_index=-100, n_docs=None, retrieval_labels=None,
    ):
        """Compute loss

        Args:
            seq_logits (_type_): _description_
            doc_scores (_type_): tensor
            target (_type_): _description_
            reduce_loss (bool, optional): _description_. Defaults to True.
            epsilon (float, optional): _description_. Defaults to 0.0.
            exclude_bos_score (bool, optional): _description_. Defaults to False.
            ignore_index (int, optional): _description_. Defaults to -100.
            n_docs (_type_, optional): _description_. Defaults to None.
            retrieval_labels (_type_, optional): _description_. Defaults to None.

        Returns:
            EasyDict: every loss requested
        """

        loss_dict = EasyDict()
        batch_size = seq_logits.shape[0] // n_docs  
        seq_len = seq_logits.shape[1]   # seq_logits dim = (batch*n_docs, seq_len , #vocabs)
        
        shift_seq_logits = seq_logits[..., :-1, :].contiguous()
        shift_target = target[..., 1:].contiguous().view(batch_size, n_docs, -1)  # [4,9,81]

        seq_logprobs = nn.functional.log_softmax(shift_seq_logits, dim=-1).view(
            batch_size, n_docs, -1, seq_logits.size(-1))  # # batch_size x n_docs x tgt_len x vocab_size,
        
        doc_logprobs = torch.log(doc_logprobs).unsqueeze(-1).unsqueeze(
            -1)  

        log_prob_sum = seq_logprobs + doc_logprobs
        rag_logprobs = torch.logsumexp(log_prob_sum, dim=1) 

        new_target = shift_target[:, 0, :].unsqueeze(-1)  
        assert new_target.dim() == rag_logprobs.dim()
        pad_mask = new_target.eq(ignore_index)  
    
        if pad_mask.any() and ignore_index < 0:
            # fill -100 to be 0, avoid indexing error using gather
            new_target.masked_fill_(pad_mask, 0)

        # Compute NLL Loss for seq_logprobs

        ll = rag_logprobs.gather(dim=-1, index=new_target)  
        smooth_obj = rag_logprobs.sum(dim=-1, keepdim=True)

        # assert pad_mask==0
        if pad_mask.any() and ignore_index < 0:
            ll.masked_fill_(pad_mask, 0)
            smooth_obj.masked_fill_(pad_mask, 0)
        ll, smooth_obj = ll.squeeze(-1), smooth_obj.squeeze(-1)

        ll = ll.sum(1)
        smooth_obj = smooth_obj.sum(1)

        nll_loss = -ll
        smooth_loss = -smooth_obj

        if reduce_loss:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
        eps_i = epsilon / rag_logprobs.size(-1)
        loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss

        return loss
        
    
        
    @torch.no_grad()
    def generate(
            self,
            batch,
            do_sample=False,
            num_beams=5,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.0,
            length_penalty=1.0,
            num_captions=1,
            temperature=1,
            max_new_tokens=25
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
        # load drug1

        
        graphs1, graphs2, mol_prompt_tokens, drug_name1, drug_name2, output_tokens,topk_scores_numpy1,topk_scores_numpy2, gt_outputs, gt_CoTs = batch

        graph_embeds1, graph_masks1 = self.graph_encoder(graphs1) #[batchsize*topk,33, 300]
        graph_embeds1 = self.ln_graph(graph_embeds1)
        query_tokens1 = self.query_tokens.expand(graph_embeds1.shape[0], -1, -1)
        query_output1 = self.Qformer.bert(
            query_embeds=query_tokens1,
            encoder_hidden_states=graph_embeds1,
            encoder_attention_mask=graph_masks1,
            return_dict=True,
        )
        mol_tokens1 = self.opt_proj(query_output1.last_hidden_state)
        # load drug2
        graph_embeds2, graph_masks2 = self.graph_encoder(graphs2)
        graph_embeds2 = self.ln_graph(graph_embeds2)
        query_tokens2 = self.query_tokens.expand(graph_embeds2.shape[0], -1, -1)
        query_output2 = self.Qformer.bert(
            query_embeds=query_tokens2,
            encoder_hidden_states=graph_embeds2,
            encoder_attention_mask=graph_masks2,
            return_dict=True,
        )
        mol_tokens2 = self.opt_proj(query_output2.last_hidden_state)
        # concat mol1 and mol2
        mol_emb = torch.cat([mol_tokens1, mol_tokens2], dim=1)

        # the prompt insert the mol embeddings
        mol_prompt_embeds = self.opt_model.get_input_embeddings()(mol_prompt_tokens.input_ids)
        mol_prompt_embeds[mol_prompt_tokens.is_mol_token] = mol_emb.flatten(0, 1) #[batch*topk*topk 16, seq_length, voc_dim]


        ## stage1
        outputs1 = self.opt_model.generate(
            inputs_embeds=mol_prompt_embeds,
            attention_mask=mol_prompt_tokens.attention_mask,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            min_length=min_length,
            # pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_return_sequences=num_captions,
            return_dict_in_generate=True,
            output_scores=True
        )
        #print("topk_scores_numpy1",topk_scores_numpy1.shape) #[batchsize,topk]
        doc_logprobs1 = nn.functional.softmax(topk_scores_numpy1[:, :self.top_k], dim=1)  # [batchsize,topk]
        
        doc_logprobs2 = nn.functional.softmax(topk_scores_numpy2[:, :self.top_k], dim=1)

        full_doc_logprobs = torch.einsum("ab,ac->abc", doc_logprobs1, doc_logprobs2).view(doc_logprobs2.shape[0], -1)  # [4,4]
        full_doc_logprobs_flat =torch.flatten(full_doc_logprobs)
     

        log_prob_sum = outputs1['sequences_scores'] + full_doc_logprobs_flat #[4*4]
        rag_logprobs = torch.exp(log_prob_sum) #[4*4]
        rag_logprobs_reshape = rag_logprobs.view(doc_logprobs2.shape[0], -1) #[4,4]
        max_prob_index = torch.argmax(rag_logprobs_reshape,-1)

       
        ddie_text = self.opt_tokenizer.batch_decode(outputs1['sequences'], skip_special_tokens=True)

        new_ddie_text_list = [ddie_text[i:i+self.top_k*self.top_k] for i in range(0, len(ddie_text), self.top_k*self.top_k)] #[4,5]
      
        max_indices_list = max_prob_index.cpu().tolist()
        selected_seq = [sublist[i] for i, sublist in zip(max_indices_list, new_ddie_text_list)]
       
        ddie_text = [text.strip() for text in selected_seq]


        return None, ddie_text, drug_name1, drug_name2

    


@dataclass
class CausalLMOutputWithPast(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    output_emb: Optional[torch.FloatTensor] = None


class OPTForCausalLM_New(OPTForCausalLM):
    def __init__(self, config):
        super(OPTForCausalLM_New, self).__init__(config)

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # print("outputs",len(outputs[1]))
        logits = self.lm_head(outputs[0]).contiguous()
        # print("logits",logits.shape) #[4, 95, 50008], 50008 is the voc number

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            output_emb=outputs[0]
        )

    def forward_test(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # print("outputs",outputs[1].shape)
        logits = self.lm_head(outputs[0]).contiguous()
        return outputs[0]


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
                    # self.model.append(nn.BatchNorm1d(hidden_dims[i + 1]))
                    self.model.append(nn.LayerNorm(hidden_dims[i + 1]))
        self.loss_classifier = nn.BCEWithLogitsLoss()

    def forward(self, h, target):
        h = self.model(h)
        loss = self.loss_classifier(h, target)
        return loss, h


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
