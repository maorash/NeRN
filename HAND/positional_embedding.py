import numpy as np
import torch
from torch import nn

from HAND.options import EmbeddingsConfig


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
        self.embedding_fusion_mode = config.fusion_mode
        self.num_idxs = config.num_idxs
        assert self.embedding_fusion_mode in ['concat', 'sum']
        assert config.enc_levels >= 1
        self.base = config.base
        self.levels = config.enc_levels
        self.output_size = self._calculate_output_size()

    def _calculate_output_size(self):
        if self.embedding_fusion_mode == 'concat':
            return self.levels * 2 * self.num_idxs
        elif self.embedding_fusion_mode == 'sum':
            return self.levels * 2

    def forward(self, pos):
        pe_list = []
        if self.embedding_fusion_mode == 'concat':
            # Efficient implementation
            x = (torch.tensor(pos).unsqueeze(-1) * (self.base ** torch.arange(self.levels)) * np.pi)
            final_embeddings = torch.dstack([torch.sin(x), torch.cos(x)]).flatten()
        elif self.embedding_fusion_mode == 'sum':
            # Non-efficient implementation
            for p in pos:
                pe_levels = p * (self.base ** torch.arange(self.levels)) * np.pi
                # Interleaving sin and cos on pe_levels
                pe_list.append(torch.dstack([torch.sin(pe_levels), torch.cos(pe_levels)]).flatten())
            final_embeddings = torch.vstack(pe_list).sum(dim=0)
        else:
            raise NotImplementedError(f'Unsupported embedding fusion mode {self.embedding_fusion_mode}.')
        return final_embeddings
