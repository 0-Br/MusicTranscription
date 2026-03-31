import os
import shutil
import json
import argparse
import warnings
from omegaconf import OmegaConf

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from transformers import T5Config, AlbertConfig
from dataset import MidiMixIterDataset
from models.t5 import T5ForConditionalGeneration
from models.albert import AlbertForTokenClassification
from utils import get_cosine_schedule_with_warmup, get_result_dir

# warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision('medium')
pl.seed_everything(3407)


class MusicTranscriptionNet(pl.LightningModule):

    def __init__(self, config, model_config, result_dir="./results"):
        super().__init__()
        self.config = OmegaConf.load(config)
        self.result_dir = result_dir
        with open(model_config) as f:
            config_dict = json.load(f)
        if config_dict["model_type"] == "t5":
            self.model: nn.Module = T5ForConditionalGeneration(T5Config.from_dict(config_dict))
        elif config_dict["model_type"] == "albert":
            self.model: nn.Module = AlbertForTokenClassification(AlbertConfig.from_dict(config_dict))
        else:
            raise TypeError

        if self.config.get('pretrained', None) is not None:
            # print(torch.load(self.config.pretrained, map_location='cpu'))
            # self.model.load_state_dict(torch.load(self.config.pretrained, map_location='cpu'), strict=True)
            pass

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        inputs = batch['inputs']
        targets = batch['targets']
        outputs = self.forward(inputs=inputs, labels=targets)
        self.log('train_loss', outputs.loss)
        return outputs.loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        inputs = batch['inputs']
        targets = batch['targets']
        outputs = self.forward(inputs=inputs, labels=targets)
        self.log('val_loss', outputs.loss)
        return outputs.loss

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), self.config.lr)
        warmup_step = int(self.config.num_training_steps / 100)
        print('warmup step: ', warmup_step)
        schedule = {
            'scheduler': get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=warmup_step, num_training_steps=self.config.num_training_steps),
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [schedule]

    def configure_callbacks(self):
        lr_monitor = LearningRateMonitor(logging_interval='step')
        checkpoint_callback = ModelCheckpoint(dirpath=f"./{result_dir}/checkpoints", filename='{epoch}-{val_loss:.8f}', monitor='val_loss', mode='min', save_last=True, save_top_k=8, save_weights_only=False)
        earlystop_callback = EarlyStopping(monitor='val_loss', patience=25, mode='min')
        return [lr_monitor, checkpoint_callback, earlystop_callback]

    def train_dataloader(self):
        train_path = self.config.data.train_path
        dataset = MidiMixIterDataset(train_path, "train", is_train=True, mel_length=self.config.mel_length, event_length=self.config.event_length, **self.config.data.config)
        train_loader = DataLoader(dataset, batch_size=self.config.per_device_batch_size, num_workers=self.config.num_workers)
        return train_loader

    def val_dataloader(self):
        test_path = self.config.data.test_path
        dataset = MidiMixIterDataset(test_path, "validation", is_train=False, mel_length=self.config.mel_length, event_length=self.config.event_length, **self.config.data.config)
        val_loader = DataLoader(dataset, batch_size=self.config.per_device_batch_size, num_workers=self.config.num_workers)
        return val_loader


def main(config, model_config, result_dir):

    model = MusicTranscriptionNet(config, model_config, result_dir)
    config = OmegaConf.load(config)
    print("Model Architecture:")
    print(model)

    logger = TensorBoardLogger(save_dir='/'.join(result_dir.split('/')[:-1]), name=result_dir.split('/')[-1])

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=config.devices,
        logger=logger,
        precision=32,
        max_steps=int(config['num_training_steps']),
        accumulate_grad_batches=config.grad_accum,
        default_root_dir="./results"
    )

    # trainer.fit(model)
    trainer.fit(model, ckpt_path="results/007/checkpoints/epoch=44-val_loss=3.71968579.ckpt")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Music Transcription')
    parser.add_argument('--config', type=str, default='./config.yaml')
    parser.add_argument('--model', type=str, default='Albert') # T5
    args = parser.parse_args()
    print('Args in Experiment:')
    print(args)

    config = args.config
    model = args.model
    model_config = f'./config/{args.model}.json'

    result_dir = get_result_dir()
    os.makedirs(result_dir, exist_ok=False)
    shutil.copy(config, f"./{result_dir}/config.yaml")
    shutil.copy(model_config, f"./{result_dir}/{args.model}.json")
    print("Save Results to:", result_dir)

    main(config, model_config, result_dir)
