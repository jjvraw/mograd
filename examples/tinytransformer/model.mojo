from mograd import Tensor
import mograd.nn as nn


struct TinyAttention:
    var n_heads: Int
    var d_head: Int

    var q: nn.Linear
    var k: nn.Linear
    var v: nn.Linear
    var proj: nn.Linear

    def __init__(out self, d_model: Int, n_heads: Int):
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, d_model)

    def __call__(self, x: Tensor) -> Tensor:
        B = x.shape(0)
        T = x.shape(1)
        C = x.shape(2)

        q = self.q(x).reshape(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k(x).reshape(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v(x).reshape(B, T, self.n_heads, self.d_head).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / (self.d_head**0.5)  # (B, nh, T, T)
