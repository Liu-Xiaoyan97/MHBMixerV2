from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

class MHBAMixerV2Config(PretrainedConfig):
    model_type = "MHBAMixerV2_LightningLM"

    def __init__(self, 
                tokenizer_name: str="meta-llama/Llama-2-7b-hf",
                vocab_size: int=32000,
                n_layers: int=12,
                embedding_dim: int=300,
                hidden_dim: int=1024, 
                n_heads: int=16,
                drop_rate: float=0.5,
                bos_token_id: int=1,
                eos_token_id: int=2,
                pad_token_id: int=2,
                batch_size: int=64,
                num_workers: int=16,
                lr: float=6e-4,
                weight_decy: float=0.1,
                log_every_n_steps: int=10,
                accelerator: str="auto",
                devices: int=1,
                max_epochs: int=10,
                **kwargs
    ):
        self.tokenizer_name = tokenizer_name
        self.vocab_size = vocab_size
        self.n_layers =  n_layers
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.drop_rate = drop_rate
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.optimizer = {
            "lr": lr,
            "weight_decay": weight_decy
        }
        self.trainer = {
            "log_every_n_steps": log_every_n_steps,
            "accelerator": accelerator,
            "devices": devices,
            "max_epochs": max_epochs
        }

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            **kwargs,
        )