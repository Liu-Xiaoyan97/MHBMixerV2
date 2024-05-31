from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
import torch
from torch import nn as nn
from MHBAMixerV2.Mixers import MHBAMixerV2
from transformers import LlamaTokenizer as Tokenizer
from lightning import LightningModule, LightningDataModule
import os
from transformers.configuration_utils import PretrainedConfig
import torch.nn.functional as F
# from torchdata.datapipes.iter import FileOpener
torch.utils.data.datapipes.utils.common.DILL_AVAILABLE = torch.utils._import_utils.dill_available()
from zipfile import ZipFile
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, load_from_disk

hf_access_token = "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"
os.environ["HF_ACCESS_TOKEN"] = hf_access_token


class MHBAMixerV2Module(LightningModule):
    def __init__(self, config: PretrainedConfig, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = Tokenizer.from_pretrained(config.tokenizer_name if hasattr(config, "tokenizer_name") else "meta-llama/Llama-2-7b-hf", token=hf_access_token, padding_side="right")
        self.tokenizer.pad_token_id = config.pad_token_id if hasattr(config, "pad_token_id") else self.tokenizer.eos_token_id
        self.vocab_size=config.vocab_size if hasattr(config, "vocab_size") else 32000
        self.MHBAMixers = MHBAMixerV2(
            vocab_size=self.vocab_size,
            n_layers=config.n_layers if hasattr(config, "n_layers") else 12,
            embedding_dim=config.embedding_dim if hasattr(config, "embedding_dim") else 300,
            hidden_dim=config.hidden_dim if hasattr(config, "hidden_dim") else 512, 
            n_heads=config.n_heads if hasattr(config, "n_heads") else 512,
            padding_idx=self.tokenizer.pad_token_id,
            drop_rate=config.drop_rate if hasattr(config, "drop_rate") else 0.02,
            num_experts=config.num_experts if hasattr(config, "num_experts") else 10,
            topk=config.topk if hasattr(config, "topk") else 2,
        )
        self.batch_size = config.batch_size if hasattr(config, "batch_size") else 64,
        self.optimizerconfig = config.optimizer if hasattr(config, "optimizer") else {"lr": 6e-4, "weight_decay": 0.01}
        self.token_shift = nn.ConstantPad1d((-1, 1), self.tokenizer.pad_token_id)
    
    def tokenized(self, batch):
        token_seq = self.tokenizer.batch_encode_plus(batch['text'], padding="longest", return_tensors="pt")['input_ids']
        # print(token_seq)
        return token_seq

    
    def forward(self, inputs):
        return self.MHBAMixers(inputs)
    
    def training_step(self, batch, batch_idx):
        batch = self.tokenized(batch).cuda()
        output = self.forward(batch)
        target = self.token_shift(batch)
        loss = F.cross_entropy(output.view(-1, self.vocab_size), target.view(-1), ignore_index=self.tokenizer.pad_token_id, reduction='mean')
        self.log("train_loss", loss.item(), prog_bar=True, on_step=True, on_epoch=True, batch_size=len(batch), sync_dist=True)
        return loss
    
    def _shared_eval_step(self, batch, batch_idx):
        batch = self.tokenized(batch).cuda()
        output = self.forward(batch)
        target = self.token_shift(batch)
        loss = F.cross_entropy(output.view(-1, 32000), target.view(-1), ignore_index=self.tokenizer.pad_token_id, reduction='mean')
        logits = torch.argmax(output, dim=-1)
        # predict_vocab = self.tokenizer.decode(logits[0])
        # print(predict_vocab)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self._shared_eval_step(batch, batch_idx)
        metrics = {'val_loss': loss.item()}
        self.log_dict(metrics, batch_size=len(batch), sync_dist=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss = self._shared_eval_step(batch, batch_idx)
        metrics = {'test_loss': loss.item()}
        self.log_dict(metrics, batch_size=len(batch), sync_dist=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.MHBAMixers.parameters(), **self.optimizerconfig)


# Guidence of LightningDataModule    
# https://lightning.ai/docs/pytorch/stable/data/datamodule.html#why-do-i-need-a-datamodule
class Text8DataModule(LightningDataModule):
    def __init__(self, 
                 config: PretrainedConfig
                 ) -> None:
        super().__init__()
        self.batch_size = config.batch_size if hasattr(config, "batch_size") else 64
        self.num_workers = config.num_workers if hasattr(config, "num_workers ") else 31
    
    # datasets need to be fixed
    def prepare_data(self) -> None:
        load_dataset("afmck/text8-chunked1024")
    
    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = load_dataset("afmck/text8-chunked1024", split="train")
            self.val_dataset = load_dataset("afmck/text8-chunked1024", split="validation")
        if stage == "test":
            self.test_dataset = load_dataset("afmck/text8-chunked1024", split="test")
    
    def train_dataloader(self) -> torch.Any:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def val_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def test_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    
