import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torch.optim as optim
from torch import nn
from torch.utils.data import Dataset, DataLoader
import gzip, time, random, math
from collections import OrderedDict
import re
from numpy.random import default_rng
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.plugins import DDPPlugin

from model import *
from utils import *


params = {
    'SRC_N_TOKENS': len(IUPAC_VOCAB),
    'EMB_SIZE': 150,
    'NHEAD': 12,
    'FFN_HID_DIM': 768,
    'MLP_DIM': 128,
    'BATCH_SIZE': 128,
    'NUM_LAYERS': 12,
    'N_EPOCHS': 100,
    'LR': 1e-4,
    'WARMUP_STEPS': 16000,
    'LOG_INTERVAL': 10,
    'OAS_FILE_LOC_VAL': "oas_bert_processed_heavy_val_100k"
}


if __name__ == '__main__':
    torch.manual_seed(0)

    print(params)

    run_name = "bert-ds-base"

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="bert-ds/checkpoints",
        filename="checkpoint-%s-{epoch:02d}-{val_loss:.2f}" % (run_name),
        save_top_k=3,
        mode="min",
        save_weights_only=False
    )

    lr_monitor_callback = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(gpus=8, accelerator="ddp",
        max_epochs=params["N_EPOCHS"],
        logger=TensorBoardLogger('bert-ds/lightning_logs/', name=run_name),
        callbacks=[checkpoint_callback, lr_monitor_callback]
        )

    model = BERTmlm(vocab_size=params['SRC_N_TOKENS'], hidden=params['FFN_HID_DIM'], mlp_dim=params['MLP_DIM'], embed_size=params["EMB_SIZE"],
        n_layers=params['NUM_LAYERS'], attn_heads=params['NHEAD'], dropout=0.1, lr=params['LR'], warmup_steps=params['WARMUP_STEPS'])

    # Initialize parameters with xavier uniform.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    dataset = oasDatasetIterable(params['EMB_SIZE'], 20000000, task='mlm')
    dataloader = DataLoader(dataset, batch_size=params['BATCH_SIZE'], num_workers=0)

    dataset_val = oasDataset(params["OAS_FILE_LOC_VAL"], params['EMB_SIZE'], 100000, task='mlm')
    dataloader_val = DataLoader(dataset_val, batch_size=params['BATCH_SIZE'], num_workers=0)

    trainer.fit(model, dataloader, dataloader_val)

