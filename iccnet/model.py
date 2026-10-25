"""Coordinate networks within LittleCMS's limits, and their export to profile stages.

(x, y, z) -> sRGB-encoded color. Hidden layers are at most 14 wide: the export adds a fifteenth
channel, a gate that paints the plane z = 0 black (see icc.gate). Activations, each an exact ICC
segmented curve:
    relu  two formula segments
    sin   one sampled segment (16,384 samples over the range the pre-activations reach)
    pwl   a learned piecewise-linear function per neuron: one sampled segment, with formula
          segments that continue the end slopes
The output goes through a sigmoid (a sampled segment) so every color is in range.
"""
import ast

import numpy as np
import torch

from . import icc


class PWL(torch.nn.Module):
    """A piecewise-linear function per channel: values at k even knots on [lo, hi], linear beyond."""

    def __init__(self, ch, k, lo=-4.0, hi=4.0):
        super().__init__()
        self.lo, self.hi, self.k = lo, hi, k
        self.v = torch.nn.Parameter(torch.relu(torch.linspace(lo, hi, k)).repeat(ch, 1))

    def forward(self, x):
        t = (x - self.lo) / (self.hi - self.lo) * (self.k - 1)
        i = t.floor().clamp(0, self.k - 2).long()
        f = t - i
        ch = torch.arange(x.shape[-1]).expand_as(i)
        v0, v1 = self.v[ch, i], self.v[ch, i + 1]
        return v0 + (v1 - v0) * f

    def curves(self, zlo=None, zhi=None):
        """As profile curves, linear out to the bounds zlo, zhi of each channel's input (if given)."""
        V = self.v.detach().double().numpy()
        if zlo is None:
            return [icc.piecewise_linear(self.lo, self.hi, v) for v in V]
        return [icc.piecewise_linear(self.lo, self.hi, v, (min(a, self.lo) - 1, max(b, self.hi) + 1))
                for v, a, b in zip(V, zlo, zhi)]

    def bounds(self, zlo, zhi):
        """Bounds of each channel's output for inputs in [zlo, zhi]: at the ends and the knots between."""
        V = self.v.detach().double().numpy()
        knots = np.linspace(self.lo, self.hi, self.k)
        out_lo, out_hi = np.empty(len(V)), np.empty(len(V))
        for c, (v, a, b) in enumerate(zip(V, zlo, zhi)):
            xs = np.concatenate([[a, b], knots[(knots > a) & (knots < b)]])
            ys = self(torch.tensor(np.broadcast_to(xs[:, None], (len(xs), len(V))).copy()))[:, c].detach().double().numpy()
            out_lo[c], out_hi[c] = ys.min(), ys.max()
        return out_lo, out_hi


class Net(torch.nn.Module):
    def __init__(self, act='pwl', width=14, depth=8, knots=64, omega=30.0):
        super().__init__()
        assert width < icc.MAX_CHANNELS, 'LittleCMS reads at most 15 channels, and the gate takes one'
        self.config = dict(act=act, width=width, depth=depth, knots=knots, omega=omega)
        dims = [3] + [width] * depth + [3]
        self.lin = torch.nn.ModuleList(torch.nn.Linear(a, b) for a, b in zip(dims, dims[1:]))
        self.act = act
        if act == 'sin':                     # SIREN initialisation, omega folded into the weights
            with torch.no_grad():
                for j, layer in enumerate(self.lin[:-1]):
                    bound = 1 / layer.in_features if j == 0 else np.sqrt(6 / layer.in_features) / omega
                    layer.weight.uniform_(-bound, bound).mul_(omega)
                    layer.bias.uniform_(-1, 1)
        if act == 'pwl':
            self.pwl = torch.nn.ModuleList(PWL(width, knots) for _ in range(depth))

    def forward(self, x, pre=None):
        for j, layer in enumerate(self.lin[:-1]):
            x = layer(x)
            if pre is not None:
                pre.append(x.detach())
            x = torch.relu(x) if self.act == 'relu' else torch.sin(x) if self.act == 'sin' else self.pwl[j](x)
        return torch.sigmoid(self.lin[-1](x))

    def stages(self, x):
        """The network as profile stages, from inputs to sRGB-encoded output (no PCS tail), with
        the gate channel added (icc.gate). x: the inputs the network will see, to size the sine's
        sampled range. Learned curves stay linear over the bounds of their inputs for any input in
        [0, 1]^3 (interval arithmetic), so they are exact for every carrier."""
        pre = []
        with torch.no_grad():
            self(torch.as_tensor(x, dtype=torch.float32), pre)
        span = max(float(p.abs().max()) for p in pre) * 1.25 + 1
        st = []
        lo, hi = np.zeros(3), np.ones(3)
        for j, layer in enumerate(self.lin):
            W, b = layer.weight.detach().double().numpy(), layer.bias.detach().double().numpy()
            st.append(('mat', W, b))
            if j == len(self.lin) - 1:
                break
            zlo = np.maximum(W, 0) @ lo + np.minimum(W, 0) @ hi + b
            zhi = np.maximum(W, 0) @ hi + np.minimum(W, 0) @ lo + b
            w = layer.out_features
            if self.act == 'relu':
                st.append(('cur', [icc.relu()] * w))
                lo, hi = np.maximum(zlo, 0), np.maximum(zhi, 0)
            elif self.act == 'sin':
                st.append(('cur', [icc.sampled(np.sin, -span, span, 16384)] * w))
                lo, hi = np.full(w, -1.0), np.full(w, 1.0)
            else:
                st.append(('cur', self.pwl[j].curves(zlo, zhi)))
                with torch.no_grad():
                    lo, hi = self.pwl[j].bounds(zlo, zhi)
        st.append(('cur', [icc.sampled(lambda t: 1 / (1 + np.exp(-t)), -12, 12, 4096)] * 3))
        return icc.gate(st)


def save(net, path, **meta):
    np.savez(path, **{k: v.detach().numpy() for k, v in net.state_dict().items()},
             __config__=np.array(repr({**net.config, **meta})))


def load(path):
    z = np.load(path)
    cfg = ast.literal_eval(str(z['__config__']))
    net = Net(**{k: cfg[k] for k in ('act', 'width', 'depth', 'knots', 'omega')})
    net.load_state_dict({k: torch.tensor(z[k]) for k in z.files if k != '__config__'})
    return net, cfg
