from contextlib import contextmanager
import numpy as np
import torch
from torch import nn

from NeRN.options import EmbeddingsConfig


def get_positional_encoding_embedder(multires, enable_encoding=True):
    if enable_encoding is False:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos]
    }

    embedder_obj = PositionalEmbedding(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.output_dim


class PositionalEmbedding:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x: p_fn(x * freq))
                out_dim += d

        self.embedding_funcs = embed_fns
        self.output_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embedding_funcs], -1)


class MyPositionalEncoding(nn.Module):
    def __init__(self, config: EmbeddingsConfig):
        super(MyPositionalEncoding, self).__init__()
        self.type = config.type
        self.embedding_fusion_mode = config.fusion_mode
        self.normalization_mode = config.normalization_mode
        self.num_idxs = config.num_idxs
        self.base = config.base
        self.levels = config.enc_levels
        self.gauss_scale = config.gauss_scale
        self.output_size = self._calculate_output_size()
        with zero_seed():
            self.ffn_B = nn.Parameter(torch.randn((self.levels * self.num_idxs, self.num_idxs))
                                      * torch.Tensor(self.gauss_scale), requires_grad=False)

    def _calculate_output_size(self):
        if self.embedding_fusion_mode == 'concat':
            return self.levels * 2 * self.num_idxs
        elif self.embedding_fusion_mode == 'sum' and self.type == 'basic':
            return self.levels * 2

    def forward(self, pos):
        if self.type == 'basic':
            return self._basic(pos)
        elif self.type == 'ffn':
            return self._ffn(pos)
        else:
            raise NotImplementedError(f'Unsupported positional embedding type {self.type}')

    def _ffn(self, pos):
        if self.embedding_fusion_mode == 'concat':
            # Efficient implementation
            x = (torch.tensor(pos) * 2 * np.pi) @ self.ffn_B.T
            final_embeddings = torch.dstack([torch.sin(x), torch.cos(x)]).flatten()
        else:
            raise NotImplementedError(f'Unsupported embedding fusion mode {self.embedding_fusion_mode} for type {self.type}')
        return final_embeddings

    def _basic(self, pos):
        if self.embedding_fusion_mode == 'concat':
            # Efficient implementation
            x = (torch.tensor(pos).unsqueeze(-1) * (self.base ** torch.arange(self.levels)) * np.pi)
            final_embeddings = torch.dstack([torch.sin(x), torch.cos(x)]).flatten()
        elif self.embedding_fusion_mode == 'sum':
            pe_list = []
            # Non-efficient implementation
            for p in pos:
                pe_levels = p * (self.base ** torch.arange(self.levels)) * np.pi
                # Interleaving sin and cos on pe_levels
                pe_list.append(torch.dstack([torch.sin(pe_levels), torch.cos(pe_levels)]).flatten())
            final_embeddings = torch.vstack(pe_list).sum(dim=0)
        else:
            raise NotImplementedError(f'Unsupported embedding fusion mode {self.embedding_fusion_mode} for type {self.type}')
        return final_embeddings

    def __hash__(self):
        pe_type = {
            'basic': 0,
            'ffn': 1
        }
        return hash((self.base, self.levels, self.num_idxs, self.output_size, pe_type[self.type],
                     *tuple(self.gauss_scale)))


@contextmanager
def zero_seed():
    seed = 0
    try:
        seed = torch.random.get_rng_state()
        torch.manual_seed(0)
        yield
    finally:
        torch.random.set_rng_state(seed)
