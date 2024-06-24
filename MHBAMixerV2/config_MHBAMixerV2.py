from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

class MHBAMixerV2Config(PretrainedConfig):
    model_type = "MHBAMixerV2_LightningLM"

    def __init__(self, 
                tokenizer_name: str="/share/home/liuxiaoyan/meta-llama/Llama-2-7b-hf",
                vocab_size: int=30522,
                n_layers: int=12,
                embedding_dim: int=384,
                hidden_dim: int=384, 
                n_heads: int=16,
                drop_rate: float=0.5,
                bos_token_id: int=1,
                eos_token_id: int=2,
                pad_token_id: int=2,
                batch_size: int=8,
                num_workers: int=2,
                lr: float=6e-4,
                weight_decy: float=0.1,
                log_every_n_steps: int=10,
                val_check_interval: float=0.01,
                check_val_every_n_epoch: int=1,
                accelerator: str="auto",
                devices: int=3,
                max_epochs: int=1000,
                activation: str="silu",
                num_experts: int=5,
                topk: int=2,
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
        self.activation = activation
        self.num_experts = num_experts,
        self.topk = topk
        self.optimizer = {
            "lr": lr,
            "weight_decay": weight_decy
        }
        self.trainer = {
            "log_every_n_steps": log_every_n_steps,
            "accelerator": accelerator,
            "devices": devices,
            "max_epochs": max_epochs,
            "val_check_interval": val_check_interval,
            "check_val_every_n_epoch": check_val_every_n_epoch
        }

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            **kwargs,
        )
