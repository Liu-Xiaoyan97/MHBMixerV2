import sys
sys.path.append('./')
import torch
import torch.nn as nn
import torch.nn.functional as F
from MHBAMixerV2.Mixers import MHBAMixerV2
from transformers import AutoTokenizer
from transformers.configuration_utils import PretrainedConfig
from MHBAMixerV2.config_MHBAMixerV2 import MHBAMixerV2Config
import argparse
from transformers import AutoTokenizer



@staticmethod
def beam_search(inputs: torch.Tensor, top_k: int, top_p: float, temperature: float):
    inputs = F.softmax(inputs.view(-1))
    inputs_values, inputs_index = torch.topk(inputs, top_k)
    inputs_values = F.softmax(inputs_values).cumsum(-1)
    index = torch.nonzero(inputs_values > top_p, as_tuple=False)[0]
    top_p_values = inputs_values[: index]
    top_p_index = inputs_index[:index]
    temperature_numerator = torch.exp(top_p_values/temperature)
    temperature_denominator = temperature_numerator.sum(-1)
    temperature_norm_res = temperature_numerator/temperature_denominator
    index = torch.multinomial(temperature_norm_res, 1, replacement=False)
    sample_res_index = top_p_index[index]
    return torch.LongTensor(sample_res_index)


class MHBAMixerV2ForGeneration(nn.Module):
    def __init__(self, config: PretrainedConfig, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
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
    
    @torch.inference_mode()
    def predict_step(self, inputs, max_length, top_k, top_p, temperature):
        input_ids = torch.LongTensor(inputs["input_ids"])
        for i in range(max_length):
            output = self.forward(input_ids)[-1]
            next_token = beam_search(output, top_k, top_p, temperature)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            if next_token.item == 102:
                break
        return input_ids
    

M2C = MHBAMixerV2Config()
M2G = MHBAMixerV2ForGeneration(M2C)

tokenizer = AutoTokenizer.from_pretrained('/share/home/liuxiaoyan/BAAI/bge-small-en-v1.5')

def predict(ckpt, inputs, max_length, top_k, top_p, temperature):
    checkpoint = torch.load(ckpt, map_location=torch.device('cpu'))
    model_weights = checkpoint["state_dict"]
    M2G.load_state_dict(model_weights)
    M2G.eval()
    input_ids = tokenizer(inputs, add_special_tokens=False)
    outputs = M2G.predict_step(input_ids, max_length, top_k, top_p, temperature)
    outputs_sentence = tokenizer.decode(outputs)
    return outputs_sentence



