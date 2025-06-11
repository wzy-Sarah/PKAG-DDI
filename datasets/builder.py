
#from .main_rag_mecddi_dataset import RAGMecDDIModule
from .main_mecddi_or_ddinter_dataset import MainDataset

def build_dataset(dataset_cfg, args, tokenizer):
    dataset_name = dataset_cfg["name"]
    #if dataset_cfg["name"]=='MecDDI_RAG':
    #    return RAGMecDDIModule(args, tokenizer, dataset_cfg)
    if dataset_cfg["name"]=='mecddi_or_ddinter':
        return MainDataset(args, tokenizer, dataset_cfg)
