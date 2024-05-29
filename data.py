from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
import torch
from torch import nn as nn
from MHBAMixerV2.Mixers import MHBAMixerV2
from transformers import LlamaTokenizer as Tokenizer
from lightning import LightningModule, LightningDataModule
import os
import torch.nn.functional as F
# from torchdata.datapipes.iter import FileOpener
torch.utils.data.datapipes.utils.common.DILL_AVAILABLE = torch.utils._import_utils.dill_available()
from zipfile import ZipFile
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, load_from_disk

hf_access_token = "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"
os.environ["HF_ACCESS_TOKEN"] = hf_access_token

tokenizer = Tokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token=hf_access_token, padding_side="right")
tokenizer.pad_token = tokenizer.eos_token

seq = ["aaaa", "hhh", "hello how are you"]
outputs = tokenizer.batch_encode_plus(seq, padding="longest", return_tensors="pt")['input_ids']
print(outputs)
print(tokenizer.decode)