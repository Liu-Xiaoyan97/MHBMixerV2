from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
import torch
from torch import nn as nn
from MHBAMixerV2.Mixers import MHBAMixerV2
from transformers import AutoTokenizer
from lightning import LightningModule, LightningDataModule
import os
from transformers.configuration_utils import PretrainedConfig
import torch.nn.functional as F
# from torchdata.datapipes.iter import FileOpener
torch.utils.data.datapipes.utils.common.DILL_AVAILABLE = torch.utils._import_utils.dill_available()
from zipfile import ZipFile
from torch.utils.data import DataLoader, Dataset, ConcatDataset, IterableDataset
from datasets import load_dataset, load_from_disk
import requests
import pickle
import json

hf_access_token = "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"
os.environ["HF_ACCESS_TOKEN"] = hf_access_token


class MHBAMixerV2Module(LightningModule):
    def __init__(self, config: PretrainedConfig, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        # self.tokenizer = Tokenizer.from_pretrained(config.tokenizer_name if hasattr(config, "tokenizer_name") else "meta-llama/Llama-2-7b-hf", token=hf_access_token, padding_side="right")
        self.vocab_size=config.vocab_size if hasattr(config, "vocab_size") else 32000
        self.MHBAMixers = MHBAMixerV2(
            n_layers=config.n_layers if hasattr(config, "n_layers") else 12,
            hidden_dim=config.hidden_dim if hasattr(config, "hidden_dim") else 512, 
            n_heads=config.n_heads if hasattr(config, "n_heads") else 512,
            drop_rate=config.drop_rate if hasattr(config, "drop_rate") else 0.02,
            activation=config.activation if hasattr(config, "activation") else "silu",
            num_experts=config.num_experts if hasattr(config, "num_experts") else 10,
            topk=config.topk if hasattr(config, "topk") else 2,
        )
        self.batch_size = config.batch_size if hasattr(config, "batch_size") else 64,
        self.optimizerconfig = config.optimizer if hasattr(config, "optimizer") else {"lr": 6e-4, "weight_decay": 0.01}
        self.token_shift = nn.ConstantPad1d((-1, 1), 102)
        self.embeddings = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.bottleneck = nn.Linear(config.embedding_dim, config.hidden_dim)
        self.postNorm = nn.LayerNorm(config.embedding_dim)
        self.llmhead = nn.Linear(config.hidden_dim, config.vocab_size)

    
    def forward(self, inputs):
        inputs = self.postNorm(self.embeddings(inputs))
        inputs = self.postNorm(self.bottleneck(inputs))
        outputs = self.MHBAMixers(inputs)
        llm_outputs = self.llmhead(outputs)
        return llm_outputs
    
    def training_step(self, batch, batch_idx):
        token_ids = batch["input_ids"]
        output = self.forward(token_ids)
        target = self.token_shift(token_ids)
        loss = F.cross_entropy(output.view(-1, self.vocab_size), target.view(-1), ignore_index=102, reduction='mean')
        self.log("train_loss", loss.item(), prog_bar=True, on_step=True, on_epoch=True, batch_size=len(batch), sync_dist=True)
        return loss
    
    def _shared_eval_step(self, batch, batch_idx):
        token_ids = batch["input_ids"]
        output = self.forward(token_ids)
        target = self.token_shift(token_ids)
        loss = F.cross_entropy(output.view(-1, 30522), target.view(-1), ignore_index=102, reduction='mean')
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
        return torch.optim.AdamW(self.parameters(), **self.optimizerconfig)


# Guidence of LightningDataModule    
# https://lightning.ai/docs/pytorch/stable/data/datamodule.html#why-do-i-need-a-datamodule
class MHBAMixerV2DataModule(LightningDataModule):
    def __init__(self, 
                 config: PretrainedConfig
                 ) -> None:
        super().__init__()
        self.batch_size = config.batch_size if hasattr(config, "batch_size") else 32
        self.num_workers = config.num_workers if hasattr(config, "num_workers ") else 31
    
    def prepare_data(self) -> None:
        load_dataset("/share/home/liuxiaoyan/afmck/text8-chunked1024")
        load_dataset("/share/home/liuxiaoyan/liuyanchen1015/VALUE_wikitext103_been_done")
        
    
    def setup(self, stage: str):
        if stage == "fit":
            train_a = Text8Dataset('train')
            train_b = Wikitext103Dataset('train')
            train_c = BookcorpusDataset('train')
            self.train_dataset = ConcatDataset([train_a, train_b, train_c])
            # self.train_dataset = train_b
            self.val_dataset = Text8Dataset("validation")
        if stage == "test":
            self.test_dataset = load_dataset("/share/home/liuxiaoyan/afmck/text8-chunked1024", split="test")
    
    def train_dataloader(self) -> torch.Any:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def val_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def test_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

### 需要传递token id 再使用gunicorn重新部署flask

def tokenizeAndembedding(field):
    headers = headers = {
    'Content-Type': 'application/json'
}
    response = requests.post('http://127.0.0.1:8000/embedding', json=json.dumps({'field': field}), headers=headers).content
    response = pickle.loads(response)
    return response


class MHBAMixerV2Dataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.data = None
        self.tokenizer = AutoTokenizer.from_pretrained('/share/home/liuxiaoyan/BAAI/bge-small-en-v1.5')

    def processing(self, field):
        pass

    def __getitem__(self, index):
        return self.processing(self.data[index])

    def __len__(self):
        return len(self.data)


class Text8Dataset(MHBAMixerV2Dataset):
    def __init__(self, mode) -> None:
        super().__init__()
        self.data = load_dataset("/share/home/liuxiaoyan/afmck/text8-chunked1024", split=mode)
    
    def processing(self, field):
        field = self.tokenizer(text=field['text'], padding="max_length", max_length=512, truncation=True, return_tensors="pt")
        return field
    
class Wikitext103Dataset(MHBAMixerV2Dataset):
    def __init__(self, mode) -> None:
        super().__init__()
        self.data = load_dataset("/share/home/liuxiaoyan/liuyanchen1015/VALUE_wikitext103_been_done", split=mode)

    def processing(self, field):
        field = self.tokenizer(text=field['sentence'], padding="max_length", max_length=512, truncation=True, return_tensors="pt")
        return field
    
class BookcorpusDataset(MHBAMixerV2Dataset):
    def __init__(self, mode) -> None:
        super().__init__()
        self.data = load_dataset("/share/home/liuxiaoyan/bookcorpus", split=mode, trust_remote_code=True)
    
    def processing(self, field):
        field = self.tokenizer(text=field['text'], padding="max_length", max_length=512, truncation=True, return_tensors="pt")
        return field