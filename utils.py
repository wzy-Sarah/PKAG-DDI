import pickle
import numpy
import matplotlib.pyplot as plt
import pandas
import os
import re
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn import metrics
import random
import torch
from rouge_score import rouge_scorer
from tqdm import tqdm
from os import path as osp

def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == '':
        return
    dir_name = osp.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def do_compute_metrics(pred, target):

    assert pred.shape[0]==target.shape[0]

    acc = metrics.accuracy_score(target, pred)
    f1_score = metrics.f1_score(target, pred,average="macro")

    p = metrics.precision_score(target, pred, average="macro")
    r = metrics.recall_score(target, pred,average="macro")
    performance = {"acc":acc,"f1":f1_score,"precision":p,"recall":r}
    print("performance",performance)

    return performance



def acc_kuanfan(logit, target):
    count = 0
    for i in range(logit.shape[0]):
        log = logit[i,:]

        tar = target[i,:]
        for j in range(tar.shape[0]):
            #print(log[j],tar[j])
            if log[j]==tar[j]==1.0:
                count = count+1
                break
    assert count<=logit.shape[0]
    #print(count/logit.shape[0])
    return count/logit.shape[0]


def test_molcap( outputs,gts):

    tokenizer = BertTokenizerFast.from_pretrained(args.text2mol_bert_path)
    output_tokens = []
    gt_tokens = []
    meteor_scores = []
    rouge_scores = []
    #text2mol_scores = []
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
    # text2mol = Text2MolMLP(
    #     ninp=768,
    #     nhid=600,
    #     nout=300,
    #     model_name_or_path=args.text2mol_bert_path,
    #     cid2smiles_path=os.path.join(args.text2mol_data_path, "cid_to_smiles.pkl"),
    #     cid2vec_path=os.path.join(args.text2mol_data_path, "test.txt")
    # )
    # text2mol.load_state_dict(torch.load(args.text2mol_ckpt_path), strict=False)
    # device = torch.device(args.device)
    # text2mol.to(device)
    mkdir_or_exist(args.caption_save_path)

    with open(os.path.join(args.caption_save_path,f"{mode}_outputs.txt"), "w") as f:
        f.write("SMILES\tground truth\toutput\n")
        for i in range(len(outputs)):
            output_tokens.append(tokenizer.tokenize(outputs[i], truncation=True, max_length=512, padding='max_length'))
            output_tokens[i] = list(filter(('[PAD]').__ne__, output_tokens[i]))
            output_tokens[i] = list(filter(('[CLS]').__ne__, output_tokens[i]))
            output_tokens[i] = list(filter(('[SEP]').__ne__, output_tokens[i]))

            gt_tokens.append(tokenizer.tokenize(gts[i], truncation=True, max_length=512, padding='max_length'))
            gt_tokens[i] = list(filter(('[PAD]').__ne__, gt_tokens[i]))
            gt_tokens[i] = list(filter(('[CLS]').__ne__, gt_tokens[i]))
            gt_tokens[i] = [list(filter(('[SEP]').__ne__, gt_tokens[i]))]

            meteor_scores.append(meteor_score(gt_tokens[i], output_tokens[i]))
            rouge_scores.append(scorer.score(outputs[i], gts[i]))
            #text2mol_scores.append(text2mol(test_dataset.smiles[i], outputs[i], device).detach().cpu().item())
            f.write(test_dataset.twodurgid[i] + '\t' + gts[i] + '\t' + outputs[i] + '\n')
    bleu2 = corpus_bleu(gt_tokens, output_tokens, weights=(0.5, 0.5))
    bleu4 = corpus_bleu(gt_tokens, output_tokens, weights=(0.25, 0.25, 0.25, 0.25))

    return {
        "BLEU-2": bleu2,
        "BLEU-4": bleu4,
        "Meteor": np.mean(meteor_scores),
        "ROUGE-1": np.mean([rs['rouge1'].fmeasure for rs in rouge_scores]),
        "ROUGE-2": np.mean([rs['rouge2'].fmeasure for rs in rouge_scores]),
        "ROUGE-L": np.mean([rs['rougeL'].fmeasure for rs in rouge_scores])
        #"Text2Mol": np.mean(text2mol_scores)
    }
def make_label_vector(labels, num_classes):

    label_vector = torch.zeros(len(labels), num_classes)

    for i, label in enumerate(labels):
        label_vector[i, label] = 1  
    return label_vector

def results_metrics(prediction=None, target=None):
    with open("./data/MecDDI/mechanism_des2id.pkl","rb") as f:
        ddie2id = pickle.load(f)
    with open("./data/MecDDI/id2mechanism.pkl","rb") as f:
        id2ddie = pickle.load(f)

    all_ddie_descriptions = []
    for id in range(len(ddie2id)):
        all_ddie_descriptions.append(id2ddie[id])
    gt_all_ddie_num = len(all_ddie_descriptions)

    all_text = all_ddie_descriptions+prediction
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(all_text)
    cosine_sim = cosine_similarity(X, X) #[103+1050,103+1050]
    pre = cosine_sim[gt_all_ddie_num:,:gt_all_ddie_num]
    pred = np.argmax(pre, axis=1)

    gt_label = []
    for p in target:     
        id = ddie2id[p[:-1]]
        gt_label.append(id)
    gt_label = np.array(gt_label)
    print("gt_label:",len(gt_label))
    perfor = do_compute_metrics(pred, gt_label)
    print("performance:", perfor)
    return perfor



def caption_evaluate(predictions, targets, tokenizer, text_trunc_length):
    meteor_scores = []
    references = []
    hypotheses = []
    for gt, out in tqdm(zip(targets, predictions)):
        gt_tokens = tokenizer.tokenize(gt, truncation=True, max_length=text_trunc_length,
                                            padding='max_length')
        gt_tokens = list(filter(('[PAD]').__ne__, gt_tokens))
        gt_tokens = list(filter(('[CLS]').__ne__, gt_tokens))
        gt_tokens = list(filter(('[SEP]').__ne__, gt_tokens))

        out_tokens = tokenizer.tokenize(out, truncation=True, max_length=text_trunc_length,
                                            padding='max_length')
        out_tokens = list(filter(('[PAD]').__ne__, out_tokens))
        out_tokens = list(filter(('[CLS]').__ne__, out_tokens))
        out_tokens = list(filter(('[SEP]').__ne__, out_tokens))

        references.append([gt_tokens])
        hypotheses.append(out_tokens)

        mscore = meteor_score([gt_tokens], out_tokens)
        meteor_scores.append(mscore)

    bleu2 = corpus_bleu(references, hypotheses, weights=(.5,.5))
    bleu4 = corpus_bleu(references, hypotheses, weights=(.25,.25,.25,.25))
    bleu2 *= 100
    bleu4 *= 100

    print('BLEU-2 score:', bleu2)
    print('BLEU-4 score:', bleu4)
    _meteor_score = np.mean(meteor_scores)
    _meteor_score *= 100
    print('Average Meteor score:', _meteor_score)

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])

    rouge_scores = []

    references = []
    hypotheses = []

    for gt, out in tqdm(zip(targets, predictions)):
        rs = scorer.score(out, gt)
        rouge_scores.append(rs)

    print('ROUGE score:')
    rouge_1 = np.mean([rs['rouge1'].fmeasure for rs in rouge_scores]) * 100
    rouge_2 = np.mean([rs['rouge2'].fmeasure for rs in rouge_scores]) * 100
    rouge_l = np.mean([rs['rougeL'].fmeasure for rs in rouge_scores]) * 100
    print('rouge1:', rouge_1)
    print('rouge2:', rouge_2)
    print('rougeL:', rouge_l)
    return bleu2, bleu4, rouge_1, rouge_2, rouge_l, _meteor_score



class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


if __name__ == "__main__":
    prediction = []
    results_metrics()


