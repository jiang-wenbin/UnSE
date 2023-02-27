import torch
from torch import nn

class Conv2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 2), stride=(2, 1), 
                 padding=(0, 1), d_norm='spectral_norm'):
        super().__init__()
        norm_f = getattr(torch.nn.utils, d_norm)
        self.conv = norm_f(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
        self.activation = nn.LeakyReLU(0.1)
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        x = self.conv(x)        
        x = self.activation(x)
        return x


class STFT(torch.nn.Module):
    """Speech Enhancement Base Class"""

    def __init__(self, conf):
        # fft_m: fft multipler
        super().__init__()        
        window_func = getattr(
            torch, "{}_window".format(conf.get("window", "hann")))
        window = window_func(conf.stft_kwargs['win_length'])
        self.register_buffer("window", window)

    def apply_stft(self, x):
        """
        x: shape of [n_batch, n_samples]
        return: shape of [n_batch, 1, n_fft/2+1, n_frames]
        """
        X_stft = torch.stft(x, **self.stft_kwargs,
                            window=self.window, return_complex=True)
        return torch.unsqueeze(X_stft, 1)


class Discriminator(nn.Module):
    def __init__(self, conf, param):
        super().__init__()
        self.conf = conf
        self.stft = STFT(conf)
        self.debug = conf.get('debug', False)
        d_norm = conf.get('d_norm', 'spectral_norm') # spectral_norm, weight_norm
        self.models = nn.ModuleList([Conv2DBlock(*item, d_norm=d_norm) for item in param])
        self.post_models = nn.ModuleList()
        norm_f = getattr(torch.nn.utils, d_norm) if conf.get('d_linear_norm', False) else nn.Identity()
        linear_nodes = [int(item) for item in conf.get('linear_nodes', '256,64,1').split(',')]
        self.post_models.extend([norm_f(nn.Linear(linear_nodes[0], linear_nodes[1])),
                                nn.LeakyReLU(0.1),
                                norm_f(nn.Linear(linear_nodes[1], linear_nodes[2]))])

    def forward(self, x):
        if len(x.shape) == 3 and x.shape[1] == 1:
            x = x.squeeze(1)
        spec = self.stft.apply_stft(x)
        data_list = [spec.real, spec.imag]
        data = torch.cat(data_list, dim=1)
        if self.debug:
            print(f"data: {data.shape}")
        for i, layer in enumerate(self.models):
            data = layer(data)
            if self.debug:
                print(f"layer_{i}: {data.shape}")
        data = torch.flatten(data, 1, -1)        
        for i, layer in enumerate(self.post_models):
            data = layer(data)
            if self.debug:
                print(f"layer_{i}: {data.shape}")
        return data