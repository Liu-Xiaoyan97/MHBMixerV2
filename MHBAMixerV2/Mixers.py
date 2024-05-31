import torch
from torch import nn as nn
from transformers.activations import ACT2FN


class TokenMixingMoE(nn.Module):
    def __init__(self, 
                 hidden_dim: int, 
                 internal_dim: int, 
                 activation: str, 
                 drop_rate: float, 
                 num_experts: int, 
                 k: int=2):
        super(TokenMixingMoE, self).__init__()
        self.k = k
        self.expert_size = hidden_dim
        self.experts = nn.ModuleList([
            MHBAMixerV2TokenMixer(
                hidden_dim=hidden_dim, 
                internal_dim=internal_dim, 
                activation=activation,
                drop_rate=drop_rate) for _ in range(num_experts)
            ])
        self.gate = nn.Linear(hidden_dim, num_experts)
    
    def forward(self, x):
        gate_outputs = self.gate(x)
        topk_values, topk_indices = torch.topk(gate_outputs, self.k, dim=1)

        expert_outputs = torch.zeros(x.size(0), self.expert_size).to(x.device)
        for i in range(self.k):
            expert_idx = topk_indices[:, i]
            expert_weight = topk_values[:, i].unsqueeze(1)
            expert_output = self.experts[expert_idx](x)
            expert_outputs += expert_weight * expert_output
        
        return expert_outputs
    

class MHBAMixerV2MemoryMixer(nn.Module):
    def __init__(self, hidden_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.forget_gate = nn.Parameter(torch.randn(hidden_dim), requires_grad=True)

    def forward(self, keys: torch.Tensor, values: torch.Tensor, memorys: torch.Tensor):
        current_cell = torch.mul(keys, values)+self.forget_gate*memorys
        current_memorys = (1-self.forget_gate)*current_cell + self.forget_gate*memorys
        return current_memorys


class MHBAMixerV2TokenMixer(nn.Module):
    def __init__(self, 
        hidden_dim: torch.Tensor, 
        internal_dim: torch.Tensor, 
        activation: str,
        drop_rate: float,
        *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.activate = ACT2FN[activation]
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.drop_out = nn.Dropout(drop_rate)
        self.linear1 = nn.Linear(hidden_dim, internal_dim, bias=False)
        self.ln2 = nn.LayerNorm(internal_dim)
        self.linear2 = nn.Linear(internal_dim, hidden_dim)
        
    
    def forward(self, input_: torch.Tensor):
        flatten_input_ = torch.flatten(input_, start_dim=0, end_dim=-2)
        token_mixing = self.linear1(self.activate(self.drop_out(self.ln1(flatten_input_))))
        token_mixing = self.linear2(self.activate(self.drop_out(self.ln2(token_mixing))))
        token_reconstruct = token_mixing.reshape(input_.size())
        return token_reconstruct


def fourier_transform(x, num_signal: int):
    return torch.fft.fftn(x, s=num_signal, dim=-1).real

def reverse_fourier_transform(x, num_signal:int):
    return torch.fft.ifftn(x, s=num_signal, dim=-1).real


class MHBAMixerV2Block(nn.Module):
    def __init__(self,
        hidden_dim: int,  
        n_heads: int, 
        activation: str,
        drop_rate: float=0.5, 
        num_experts: int=10,
        topk: int=2,
        *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # assert hidden_dim%n_heads, "hidden_dim must be divisible by n_heads!"
        self.head_dim = int(hidden_dim / n_heads)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.n_heads = n_heads
        
        self.tokenMixingMoE = TokenMixingMoE(hidden_dim=self.head_dim, 
                                             internal_dim=self.head_dim * 2, 
                                             activation=activation, 
                                             drop_rate=drop_rate, 
                                             num_experts=num_experts, 
                                             k=topk)
    
        self.memory_mixer = MHBAMixerV2MemoryMixer(hidden_dim=self.head_dim)
    
    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, memorys: torch.Tensor):
        bsz = queries.shape[0]
        queries_perpare = queries.view(bsz, -1, self.n_heads, self.head_dim).transpose(1, 2)
        keys_perpare = keys.view(bsz, -1, self.n_heads, self.head_dim).transpose(1, 2)
        values_perpare =values.view(bsz, -1, self.n_heads, self.head_dim).transpose(1, 2)
        
        queries_ = self.queries(queries_perpare)
        keys_ = self.keys(keys_perpare)
        values_ = self.values(values_perpare)

        current_memorys = self.memory_mixer.forward(keys_, values_, memorys)
        cell_values = torch.mul(queries_, current_memorys)
        token_mixing = self.tokenMixingMoE.forward(cell_values).transpose(1, 2)
        
        token_mixing = token_mixing.reshape(bsz, -1, self.n_heads*self.head_dim)

        return token_mixing, current_memorys


class MHBAMixerV2(nn.Module):
    def __init__(self, 
        vocab_size: int,
        n_layers: int,
        embedding_dim: int,  
        hidden_dim: int,
        n_heads: int, 
        padding_idx: int,
        activation: str,
        drop_rate: float=0.5, 
        num_experts: int=10,
        topk: int=2,
        *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx)
        self.bottleneck = nn.Linear(embedding_dim, hidden_dim)
        self.hidden_dim = hidden_dim
        self.memorys = nn.Parameter(torch.randn(hidden_dim//n_heads))
        self.MHBAMixerV2Blocks = nn.ModuleList(
            MHBAMixerV2Block(
                hidden_dim=hidden_dim, 
                n_heads=n_heads, 
                activation=activation, 
                drop_rate=drop_rate,
                num_experts=num_experts[0],
                topk=topk) for i in range(n_layers)
        )
        self.llm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
    
    def forward(self,
            inputs_: torch.Tensor):
        token_seq_embedding = self.embedding(inputs_)
        bottleneck = self.bottleneck(token_seq_embedding)
        # print(f" token embedding shape: ---->{token_seq_embedding.shape}")
        token_mixing = fourier_transform(bottleneck, self.hidden_dim)
        memory = self.memorys
        for layer in self.MHBAMixerV2Blocks:
            token_mixing, memory = layer(token_mixing, token_mixing, token_mixing, memory)
        token_mixing = reverse_fourier_transform(token_mixing, self.hidden_dim)
        # print(token_mixing.shape)
        outputs = self.llm_head(token_mixing)
        return outputs
        
