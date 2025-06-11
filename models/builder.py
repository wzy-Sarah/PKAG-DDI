#from .model_main_rag import MainModel_RAG
from .model_main_function_classification import MainModel_Function_CLS
from .model_main_function_generation import MainModel_Function_gen
def build_model(args, cfg):
    model_name = cfg["name"]
    if model_name == 'Function_cls':
        return MainModel_Function_CLS(args, cfg)
    elif model_name == 'Function_gen':
        return MainModel_Function_gen(args,cfg)