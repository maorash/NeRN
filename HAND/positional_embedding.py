import numpy as np
import torch
from torch import nn


# TODO: configure this in options and figure out what happens here...specifically `input_dims` that's 3 but can act
#  on a higher dimensional input?
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
    def __init__(self, enc_levels=80, base=1.25):
        super(MyPositionalEncoding, self).__init__()
        assert enc_levels >= 1
        self.base = base
        self.levels = enc_levels
        self.enc_len = 2 * enc_levels

    def forward(self, pos):
        pe_list = []
        # naive implementation using a simple concat
        for p in range(len(pos)):
            for i in range(self.levels):
                temp_value = torch.tensor(p * (self.base ** i) * np.pi)
                pe_list += [torch.sin(temp_value), torch.cos(temp_value)]
        return torch.stack(pe_list, 0)
