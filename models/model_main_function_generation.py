import os
import os.path as osp
from typing import Any, Dict
import torch


from .blip2_function_retrieval import Blip2OPT_RETRIEVAL,Blip2OPT_RETRIEVAL_marginalize
import pytorch_lightning as pl
from torch import optim
from lavis.common.optims import LinearWarmupCosineLRScheduler, LinearWarmupStepLRScheduler
import json
from utils import results_metrics, caption_evaluate, AttrDict, do_compute_metrics
import torch.distributed as dist
from peft import LoraConfig, TaskType
import numpy as np
from transformers import BertTokenizerFast
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import json

MODOLS = {
    "retrieval":Blip2OPT_RETRIEVAL,
    "marginalize":Blip2OPT_RETRIEVAL_marginalize
}


# peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
class MainModel_Function_gen(pl.LightningModule):  
    def __init__(self, args, cfg):
        super().__init__()

        self.args = args
        self.cfg = cfg
        self.caption_eval_epoch = cfg["caption_eval_epoch"]
        self.stage = cfg["stage"]

        # for generate text
        self.do_sample = cfg["do_sample"]
        self.num_beams = cfg["num_beams"]
        self.max_len = cfg["generate_max_len"]
        self.min_len = cfg["generate_min_len"]
        self.temperature = cfg["temperature"]
        self.top_p = cfg["top_p"]
        self.max_new_tokens = cfg["max_new_tokens"]
        self.repetition_penalty = cfg["repetition_penalty"]
        # self.use_rag = cfg["use_rag"]
        self.batch_size = cfg["batch_size"]

        # for the opt_model
        self.llm_tune = cfg["llm_tune"]  # ["lora","freeze","full"]
        self.peft_dir = cfg["peft_dir"]  # whether the peft has init checkpoint
        self.opt_model = cfg["opt_model"]  # "facebook/galactica-1.3b"
        # self.prompt = cfg["prompt"]
        if self.opt_model.find('galactica') >= 0: 
            self.blip2opt =MODOLS[self.stage](cfg)  

        elif self.opt_model.find('llama') >= 0 or self.opt_model.find('vicuna') >= 0:
            self.blip2opt = Blip2Llama(cfg["bert_name"], cfg["gin_num_layers"], cfg["gin_hidden_dim"],
                                       cfg["drop_ratio"],
                                       cfg["tune_gnn"], cfg["num_query_token"], cfg["cross_attention_freq"],
                                       self.llm_tune,
                                       self.peft_dir, self.opt_model, args=cfg)
        else:
            raise NotImplementedError()
        self.tokenizer = self.blip2opt.init_tokenizer()
        self.save_hyperparameters(args)

    def training_step(self, batch, batch_idx):

        if self.scheduler:
            self.scheduler.step(self.trainer.current_epoch, self.trainer.global_step)

        ###============== Overall Loss ===================###
        loss = self.blip2opt(batch)
        self.log("molecule loss", float(loss['loss']), batch_size=self.batch_size, sync_dist=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], batch_size=self.batch_size, sync_dist=True)
        return loss['loss']

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx=None):

        if (self.current_epoch + 1) % self.caption_eval_epoch != 0:
            return

        loss = self.blip2opt(batch[:-2])
        self.log("val_loss", float(loss['loss']), batch_size=self.batch_size, sync_dist=True)

        CoT_pred, predictions, drug1, drug2 = self.blip2opt.generate(
            batch,
            do_sample=self.do_sample,
            num_beams=self.num_beams,
            # max_length=self.max_len,
            min_length=self.min_len,
            max_new_tokens=self.max_new_tokens
        )
        # print
        return CoT_pred, predictions, batch[-2], batch[-1]

    def validation_epoch_end(self, outputs):
        if self.current_epoch != 0:
            if (self.current_epoch + 1) % self.caption_eval_epoch != 0:
                return
            caption_outputs = outputs
            CoT_pred_list, list_predictions, CoT_target_list, list_targets = zip(*caption_outputs)
            if CoT_pred_list[0] == None:
                predictions = [i for ii in list_predictions for i in ii]
                targets = [i for ii in list_targets for i in ii]
                all_predictions = [None for _ in range(self.trainer.world_size)]
                all_targets = [None for _ in range(self.trainer.world_size)]
                dist.all_gather_object(all_predictions, predictions)
                dist.all_gather_object(all_targets, targets)
                if self.global_rank == 0:
                    all_predictions = [i for ii in all_predictions for i in ii]
                    all_targets = [i for ii in all_targets for i in ii]

                    self.save_predictions_valid(all_predictions, all_targets, None, None)
            else:
                predictions = [i for ii in list_predictions for i in ii]
                targets = [i for ii in list_targets for i in ii]
                CoT_preds = [i for ii in CoT_pred_list for i in ii]
                CoT_targets = [i for ii in CoT_target_list for i in ii]
                all_predictions = [None for _ in range(self.trainer.world_size)]
                all_targets = [None for _ in range(self.trainer.world_size)]
                all_CoT_preds = [None for _ in range(self.trainer.world_size)]
                all_CoT_targets = [None for _ in range(self.trainer.world_size)]
                # all_predictions = predictions
                # all_targets =targets
                dist.all_gather_object(all_predictions, predictions)
                dist.all_gather_object(all_targets, targets)
                dist.all_gather_object(all_CoT_preds, CoT_preds)
                dist.all_gather_object(all_CoT_targets, CoT_targets)
                if self.global_rank == 0:
                    all_predictions = [i for ii in all_predictions for i in ii]
                    all_targets = [i for ii in all_targets for i in ii]
                    all_CoT_preds = [i for ii in all_CoT_preds for i in ii]
                    all_CoT_targets = [i for ii in all_CoT_targets for i in ii]
                    # print("all_targets", len(all_targets))
                    self.save_predictions_valid(all_predictions, all_targets, all_CoT_preds, all_CoT_targets)

    def save_predictions_valid(self, predictions, targets, all_CoT_preds, all_CoT_targets):

        assert len(predictions) == len(targets)
        print("****************show result*********************************")
        for j in range(5):
            print("Generated   : %s" % predictions[-j])
            print("Ground truth: %s" % targets[- j])
            print("------------------------------------------------------")
        print("************************************************************")
        if all_CoT_preds is not None:

            print("****************show CoT results*********************************")
            for j in range(5):
                print("Generated   : %s" % all_CoT_preds[-j])
                print("Ground truth: %s" % all_CoT_targets[- j])
                print("------------------------------------------------------")
            print("************************************************************")

        tokenizer = BertTokenizerFast.from_pretrained("./all_checkpoints/bert_pretrained")
        output_tokens = []
        gt_tokens = []
        meteor_scores = []
        rouge_scores = []
        #text2mol_scores = []
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])

        for i in range(len(predictions)):
            output_tokens.append(tokenizer.tokenize(predictions[i], truncation=True, max_length=512, padding='max_length'))
            output_tokens[i] = list(filter(('[PAD]').__ne__, output_tokens[i]))
            output_tokens[i] = list(filter(('[CLS]').__ne__, output_tokens[i]))
            output_tokens[i] = list(filter(('[SEP]').__ne__, output_tokens[i]))

            gt_tokens.append(tokenizer.tokenize(targets[i], truncation=True, max_length=512, padding='max_length'))
            gt_tokens[i] = list(filter(('[PAD]').__ne__, gt_tokens[i]))
            gt_tokens[i] = list(filter(('[CLS]').__ne__, gt_tokens[i]))
            gt_tokens[i] = [list(filter(('[SEP]').__ne__, gt_tokens[i]))]

            meteor_scores.append(meteor_score(gt_tokens[i], output_tokens[i])) #gt_tokens[i], output_tokens[i])
            rouge_scores.append(scorer.score(predictions[i], targets[i])) #outputs[i], gts[i]

        bleu2 = corpus_bleu(gt_tokens, output_tokens, weights=(0.5, 0.5))
        bleu4 = corpus_bleu(gt_tokens, output_tokens, weights=(0.25, 0.25, 0.25, 0.25))

        text_metric = {
        "BLEU-2": bleu2,
        "BLEU-4": bleu4,
        "Meteor": np.mean(meteor_scores),
        "ROUGE-1": np.mean([rs['rouge1'].fmeasure for rs in rouge_scores]),
        "ROUGE-2": np.mean([rs['rouge2'].fmeasure for rs in rouge_scores]),
        "ROUGE-L": np.mean([rs['rougeL'].fmeasure for rs in rouge_scores]),
        #"Text2Mol": np.mean(text2mol_scores)
        }
        print("text_metric",text_metric)

        performance = str(text_metric)
        file_path = os.path.join(self.cfg["work_dir"], "valid_logs")
        mkdir_or_exist(file_path)
        file_name = os.path.join(file_path, f"valid_performance.txt")  # the first file
        if os.path.exists(file_name):
            directory, filename = os.path.split(file_name)
            name, extension = os.path.splitext(filename)
            count = 1
            new_filename = f"{name}({count}).{extension}"
            while os.path.exists(os.path.join(directory, new_filename)):
                count += 1
                new_filename = f"{name}({count}).{extension}"
            file_name = os.path.join(directory, new_filename)
        with open(file_name, 'w') as file:
            file.write(performance)
            # file.write(text_metric)

    @torch.no_grad()
    def test_step(self, batch, batch_idx):

        CoT_pred, predictions, drug_name1, drug_name2 = self.blip2opt.generate(
            batch,
            do_sample=self.do_sample,
            num_beams=self.num_beams,
            # max_length=self.max_len,
            min_length=self.min_len,
            max_new_tokens=self.max_new_tokens
        )
        return CoT_pred, predictions, batch[-2], batch[-1],drug_name1, drug_name2


    def test_epoch_end(self, outputs):
        print("Entering test_epoch_end")

        caption_outputs = outputs
        # list_predictions, list_targets = zip(*caption_outputs)
        CoT_pred_list, list_predictions, CoT_target_list, list_targets, drug_name1, drug_name2 = zip(*caption_outputs)

        #if CoT_pred_list[0] == None:
        predictions = [i for ii in list_predictions for i in ii]
        targets = [i for ii in list_targets for i in ii]
        drug1 = [i for ii in drug_name1 for i in ii]
        drug2 = [i for ii in drug_name2 for i in ii]
        all_predictions = [None for _ in range(self.trainer.world_size)]
        all_targets = [None for _ in range(self.trainer.world_size)]
        all_drug1 =[None for _ in range(self.trainer.world_size)]
        all_drug2 = [None for _ in range(self.trainer.world_size)]
        dist.all_gather_object(all_predictions, predictions)
        dist.all_gather_object(all_targets, targets)
        dist.all_gather_object(all_drug1, drug1)
        dist.all_gather_object(all_drug2, drug2)
        if self.global_rank == 0:
            all_predictions = [i for ii in all_predictions for i in ii]
            all_targets = [i for ii in all_targets for i in ii]
            all_drug1s = [i for ii in all_drug1 for i in ii]
            all_drug2s = [i for ii in all_drug2 for i in ii]

            # print("all_targets", len(all_targets))
            self.save_predictions_test(all_predictions, all_targets, all_drug1s, all_drug2s)
        

    def save_predictions_test(self, predictions, targets, drug_name1, drug_name2):

        assert len(predictions) == len(targets)

        print("****************show result*********************")
        for j in range(5):
            print("Generated   : %s" % predictions[-j])
            print("Ground truth: %s" % targets[- j])
            print("------------------------------------------------------")
        print("*************************************************")

        file_path = os.path.join(self.cfg["work_dir"], "test_logs")
        mkdir_or_exist(file_path)
        json_dict = {}
        for p, t, d1, d2 in zip(predictions, targets, drug_name1, drug_name2):
            json_dict[d1+"&"+d2]={'prediction': p, 'target': t}

        with open(os.path.join(file_path, 'predictions.txt'), 'w') as json_file:
            json.dump(json_dict, json_file, indent=4)

        tokenizer = BertTokenizerFast.from_pretrained("./all_checkpoints/bert_pretrained")
        output_tokens = []
        gt_tokens = []
        meteor_scores = []
        rouge_scores = []
        #text2mol_scores = []
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])

        for i in range(len(predictions)):
            output_tokens.append(tokenizer.tokenize(predictions[i], truncation=True, max_length=512, padding='max_length'))
            output_tokens[i] = list(filter(('[PAD]').__ne__, output_tokens[i]))
            output_tokens[i] = list(filter(('[CLS]').__ne__, output_tokens[i]))
            output_tokens[i] = list(filter(('[SEP]').__ne__, output_tokens[i]))

            gt_tokens.append(tokenizer.tokenize(targets[i], truncation=True, max_length=512, padding='max_length'))
            gt_tokens[i] = list(filter(('[PAD]').__ne__, gt_tokens[i]))
            gt_tokens[i] = list(filter(('[CLS]').__ne__, gt_tokens[i]))
            gt_tokens[i] = [list(filter(('[SEP]').__ne__, gt_tokens[i]))]

            meteor_scores.append(meteor_score(gt_tokens[i], output_tokens[i])) #gt_tokens[i], output_tokens[i])
            rouge_scores.append(scorer.score(predictions[i], targets[i])) #outputs[i], gts[i]

        bleu2 = corpus_bleu(gt_tokens, output_tokens, weights=(0.5, 0.5))
        bleu4 = corpus_bleu(gt_tokens, output_tokens, weights=(0.25, 0.25, 0.25, 0.25))

        text_metric = {
        "BLEU-2": bleu2,
        "BLEU-4": bleu4,
        "Meteor": np.mean(meteor_scores),
        "ROUGE-1": np.mean([rs['rouge1'].fmeasure for rs in rouge_scores]),
        "ROUGE-2": np.mean([rs['rouge2'].fmeasure for rs in rouge_scores]),
        "ROUGE-L": np.mean([rs['rougeL'].fmeasure for rs in rouge_scores]),
        #"Text2Mol": np.mean(text2mol_scores)
        }
        print("text_metric",text_metric)

        performance = str(text_metric)
        file_path = os.path.join(self.cfg["work_dir"], "test_logs")
        mkdir_or_exist(file_path)
        
        with open(os.path.join(file_path, f"test_performance.txt"), 'w') as file:
            file.write(performance)

    def configure_optimizers(self):
        self.trainer.reset_train_dataloader()
        warmup_steps = min(len(self.trainer.train_dataloader), self.cfg["warmup_steps"])
        optimizer = optim.AdamW(self.parameters(), lr=self.cfg["init_lr"], weight_decay=self.cfg["weight_decay"])
        if self.cfg["scheduler"] == 'linear_warmup_cosine_lr':
            self.scheduler = LinearWarmupCosineLRScheduler(optimizer, self.cfg["max_epochs"], self.cfg["min_lr"],
                                                           self.cfg["init_lr"], warmup_steps, self.cfg["warmup_lr"])
        elif self.cfg["scheduler"] == 'linear_warmup_step_lr':
            self.scheduler = LinearWarmupStepLRScheduler(optimizer, self.cfg["max_epochs"], self.cfg["min_lr"],
                                                         self.cfg["init_lr"], self.cfg["lr_decay_rate"],
                                                         self.cfg["warmup_lr"], warmup_steps)
        elif self.cfg["scheduler"] == 'None':
            self.scheduler = None
        else:
            raise NotImplementedError()
        return optimizer

    def load_from_stage1_checkpoint(self, path):
        ckpt = torch.load(path, map_location='cpu')
        state_dict = ckpt['state_dict']
        graph_encoder_dict = get_module_state_dict(state_dict, 'blip2qformer.graph_encoder')
        qformer_dict = get_module_state_dict(state_dict, 'blip2qformer.Qformer')
        ln_graph_dict = get_module_state_dict(state_dict, 'blip2qformer.ln_graph')
        qs_weight = get_module_state_dict(state_dict, 'blip2qformer.query_tokens')
        load_ignore_unexpected(self.blip2opt.Qformer, qformer_dict)
        self.blip2opt.graph_encoder.load_state_dict(graph_encoder_dict)
        self.blip2opt.ln_graph.load_state_dict(ln_graph_dict)
        self.blip2opt.query_tokens.data.copy_(qs_weight)
        return self


def load_ignore_unexpected(model, state_dict):
    keys = set(model.state_dict().keys())
    state_dict = {k: v for k, v in state_dict.items() if k in keys}

    ## try to print keys that are not included
    model.load_state_dict(state_dict, strict=True)


def get_module_state_dict(state_dict, module_name):
    module_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(module_name):
            key = key[len(module_name) + 1:]
            if key == '':
                return value
            module_state_dict[key] = value
    return module_state_dict


def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == '':
        return
    dir_name = osp.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)


def count_subdirectories(folder_path):
    try:
        entries = os.listdir(folder_path)
        subdirectories = [entry for entry in entries if os.path.isdir(os.path.join(folder_path, entry))]
        return len(subdirectories)
    except FileNotFoundError:
        print(f"file '{folder_path}'not exsit")
        return -1  
    except Exception as e:
        print(f"error:{e}")
        return -2  