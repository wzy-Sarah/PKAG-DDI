import os
import os.path as osp
import torch
import argparse
import warnings
import pytorch_lightning as pl
from pytorch_lightning import Trainer, strategies
import pytorch_lightning.callbacks as plc
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import CSVLogger

import json
from datasets.builder import build_dataset
from models.builder import build_model
from utils import mkdir_or_exist

os.environ['OPENBLAS_NUM_THREADS'] = '0'
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
torch.set_float32_matmul_precision('medium')  # can be medium (bfloat16), high (tensorfloat32), highest (float32)

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--config', help='the config file path')
    parser.add_argument('--devices', default='0,1', help='0,1')
    parser.add_argument('--seednumber', default=42, help='number of seeds')
    parser.add_argument('--save_every_n_epochs', default=10, help='save the model every n_epochs')
    parser.add_argument('--caption_eval_epoch', type=int, default=10)
    parser.add_argument('--opt_model', type=str, default="facebook/galactica-1.3b")
    parser.add_argument('--mode', type=str, default="ft", help='pretrain or ft or breakpoint')
    parser.add_argument('--strategy_name', type=str, default=None)
    parser.add_argument('--work_dir', default=False, help='this is the saved checkpoints for test')
    parser.add_argument('--breakpoint_file', default=False, help='breakpoint retrainin')
    parser.add_argument('--peft_config', type=str, default=None)
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--precision', type=str, default='bf16')

    args = parser.parse_args()

    return args


def main(args):
    pl.seed_everything(args.seednumber)  # set seed
    args = parse_args()  # get args
    cfg = json.load(open(args.config))  # get cfg from file

    # Create output file
    work_dir = os.path.join('work_dirs', osp.splitext(osp.basename(args.config))[0])
    cfg['work_dir'] = work_dir
    cfg["model"]["work_dir"] = work_dir
    cfg["model"]["caption_eval_epoch"] = cfg["caption_eval_epoch"]

    callbacks = []  # callbacks which can save checkpoints auto and continue training
    earlystopping = EarlyStopping('molecule loss', patience=cfg["patience"],
                                  min_delta=cfg["min_delta"], mode="min")

    callbacks.append(plc.ModelCheckpoint(dirpath=work_dir,
                                         filename='{epoch:02d}',
                                         #every_n_epochs=cfg["save_every_n_epochs"],
                                         save_top_k=-1
                                         #save_last=True
                                        ))
    callbacks.append(earlystopping)

    logger = CSVLogger(save_dir=work_dir, name="logs")  

    # Get model
    if args.mode == "ft":
        model = build_model(args, cfg["model"])
        ckpt = torch.load(cfg["init_checkpoint"], map_location='cpu')
        model.load_state_dict(ckpt['state_dict'], strict=False)
        print('total params:', sum(p.numel() for p in model.parameters()))
    elif args.mode == "breakpoint":
        print("breakpoint")
        print("args.breakpoint_file", args.breakpoint_file)
        model = build_model(args, cfg["model"]).load_from_checkpoint(args.breakpoint_file, strict=False, args=args, cfg=cfg["model"])
    elif args.work_dir:
        model = build_model(args, cfg["model"]).load_from_checkpoint(args.work_dir, strict=False, args=args, cfg=cfg["model"])
        print(f"loaded init checkpoint from {args.work_dir}")

    else:
        # few shot other datasets
        model = build_model(args, cfg["model"])

    # Get dataset, becasue the tokenizer is load from model, therefore when construct the dataset, must input the tokenizer
    if args.opt_model.find('galactica') >= 0:
        tokenizer = model.blip2opt.opt_tokenizer
    elif args.opt_model.find('llama') >= 0 or args.opt_model.find('vicuna') >= 0:
        tokenizer = model.blip2opt.llm_tokenizer
    else:
        raise NotImplementedError
    datasets = build_dataset(cfg["dataset"], args, tokenizer)

    # multi device  Parallel  strategy
    if len(args.devices.split(',')) > 1:
        if args.strategy_name == 'fsdp':
            strategy = strategies.DDPFullyShardedNativeStrategy()
        elif args.strategy_name == 'deepspeed':
            strategy = strategies.DeepSpeedStrategy(stage=3)
        else:
            strategy = MyDDPSpawnStrategy(find_unused_parameters=False)
    else:
        print("oooooooooooooooooooooooooonly one gpu")
        strategy = None
        # args.devices = eval(args.devices)
        args.devices = [0]
        # args.devices = torch.device("cuda:0")

    # start to train
    if args.mode in {'pretrain', 'ft'}:
        trainer = Trainer.from_argparse_args(args,
                                             callbacks=callbacks,
                                             strategy=strategy,
                                             logger=logger
                                             )

        trainer.fit(model, datamodule=datasets)
    elif args.mode == "breakpoint":
        trainer = Trainer(resume_from_checkpoint=args.breakpoint_file,
                          callbacks=callbacks,
                          precision=args.precision, accelerator=args.accelerator)
        trainer.fit(model, datamodule=datasets)
    elif args.mode == 'eval':
        trainer = Trainer(resume_from_checkpoint=args.work_dir, precision=args.precision, accelerator=args.accelerator)
        trainer.test(model, datamodule=datasets)
    else:
        raise NotImplementedError()


class MyDDPSpawnStrategy(strategies.DDPSpawnStrategy):
    def load_model_state_dict(self, checkpoint):
        assert self.lightning_module is not None
        self.lightning_module.load_state_dict(checkpoint["state_dict"], strict=False)


if __name__ == '__main__':
    main(parse_args())

