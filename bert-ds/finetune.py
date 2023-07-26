import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torch.optim as optim
from torch import nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import gzip, time, random, math
from collections import OrderedDict
import re
from numpy.random import default_rng
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from model import *
from utils import *
from train import params



def getclassweights(dataset, n_classes):
    weights = []
    total = len(dataset)

    class_data = []
    for x in dataset:
        try:
            class_data.append(np.array(x[1], dtype=np.int))
        except:
            print(x)

    class_sum = np.count_nonzero(class_data, axis=0)
    print("Class counts: %s" % (class_sum))

    for i in range(n_classes):
        w = (1 / class_sum[i])*(total)/n_classes
        if w == np.inf:
            w = 0.
        weights.append(w)

    return weights



if __name__ == '__main__':
    torch.manual_seed(0)

    from pytorch_lightning.loggers import TensorBoardLogger
    class MyTensorBoardLogger(TensorBoardLogger):
        def save(self):
            for hparam_key, hparam_value in self.hparams.items():
                # check if hparam is child of pl.LightningModule
                if pl.LightningModule in hparam_value.__class__.__bases__:
                    self.hparams[hparam_key] = hparam_value.hparams
            super().save()

    params['BATCH_SIZE'] = 128
    params['LR'] = 1e-4
    params['N_EPOCHS'] = 100
    params['DS_CSV_FILE_LOC'] = "fc079_fc080_her2_nns_affmat_v2_paired_translated.csv"

    print(params)

    finetune_name = "bert-ds-finetune"

    checkpoint_callback_val = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints",
        filename="checkpoint-%s-{epoch:02d}-{val_loss:.2f}" % (finetune_name),
        save_top_k=3,
        mode="min",
        save_weights_only=False
    )

    checkpoint_callback_train = ModelCheckpoint(
        monitor="train_loss",
        dirpath="checkpoints",
        filename="checkpoint-%s-{epoch:02d}-{train_loss:.2f}" % (finetune_name),
        save_top_k=3,
        mode="min",
        save_weights_only=False
    )

    lr_monitor_callback = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(gpus=8, accelerator="ddp",
        gradient_clip_val=0.5,
        accumulate_grad_batches=8,
        max_epochs=params["N_EPOCHS"],
        logger=MyTensorBoardLogger('lightning_logs/', name=finetune_name),
        callbacks=[checkpoint_callback_val, checkpoint_callback_train, lr_monitor_callback],
        )


    model = BERTmlm(vocab_size=params['SRC_N_TOKENS'], hidden=params['FFN_HID_DIM'], mlp_dim=params['MLP_DIM'], embed_size=params["EMB_SIZE"],
        n_layers=params['NUM_LAYERS'], attn_heads=params['NHEAD'], dropout=0.1, lr=params['LR'], warmup_steps=params['WARMUP_STEPS'])

    model = model.load_from_checkpoint("checkpoints/checkpoint-bert-ds-base-epoch=00-val_loss=0.13.ckpt")


    dataset_bsf_nns = DSDataset(params["DS_CSV_FILE_LOC"], params['EMB_SIZE'], 500000, split='train', dataset="NNS", balance=None)
    dataset_bsf_nns_val = DSDataset(params["DS_CSV_FILE_LOC"], params['EMB_SIZE'], 500000, split='test', dataset="NNS")

    print(getclassweights(dataset_bsf_nns, 3))

    class_weights = np.array(getclassweights(dataset_bsf_nns, 3))

    dataloader_bsf = DataLoader(dataset_bsf_nns, batch_size=params['BATCH_SIZE'], num_workers=0, worker_init_fn=worker_init_fn, shuffle=True)
    dataloader_bsf_val = DataLoader(dataset_bsf_nns_val, batch_size=params['BATCH_SIZE'], num_workers=0, worker_init_fn=worker_init_fn)


    affinity_model = BERTclassifier(bert=model, lr=params['LR'])

    print("Train size: %d" % (len(dataloader_bsf.dataset)))
    print("Test size: %d" % (len(dataloader_bsf_val.dataset)))


    trainer.fit(affinity_model, dataloader_bsf, dataloader_bsf_val)

