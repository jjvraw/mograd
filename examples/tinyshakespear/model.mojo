from std.random import random_float64
from std.utils.numerics import neg_inf

from mograd import Tensor, Device
import mograd.nn as nn
from mograd.nn import Module, ModuleParam


struct TinyTransformer(Module, Movable):
    var embed: nn.Embedding
    var blocks: List[TransformerBlock]
    var lm_head: nn.Linear

    def __init__(out self, vocab_size: Int, d_model: Int, d_ff: Int, n_layers: Int, n_heads: Int):
        self.embed = nn.Embedding(vocab_size, d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.blocks = List[TransformerBlock]()
        for _ in range(n_layers):
            self.blocks.append(TransformerBlock(d_model, d_ff, n_heads))

    def __call__(mut self, x: Tensor) raises -> Tensor:
        var h = self.embed(x)
        for i in range(len(self.blocks)):
            h = self.blocks[i](h)
        return self.lm_head(h)

    def parameters(mut self) -> List[ModuleParam]:
        var ps = self.embed.parameters()
        for i in range(len(self.blocks)):
            ps += self.blocks[i].parameters()
        ps += self.lm_head.parameters()
        return ps^

    def generate(
        mut self, device: Device, context: List[Int64], max_new_tokens: Int, seq_len: Int, temperature: Float32 = 0.8
    ) raises -> List[Int64]:
        var tokens = context.copy()
        for _ in range(max_new_tokens):
            var T = min(len(tokens), seq_len)
            var ctx = List[Int64]()
            for i in range(len(tokens) - T, len(tokens)):
                ctx.append(tokens[i])
            var x = Tensor(device, ctx, (1, T))
            var logits = self(x)  # (1, T, VOCAB_SIZE)
            var last = logits.squeeze(0)[T - 1 : T]  # (1, VOCAB_SIZE)
            var probs = (last * (Float32(1.0) / temperature)).softmax().squeeze(0).to_list()
            tokens.append(Int64(sample(probs)))
        return tokens^


struct TransformerBlock(Module, Movable):
    var attn: MultiHeadAttention
    var ffn: FFN
    var norm1: nn.LayerNorm
    var norm2: nn.LayerNorm

    def __init__(out self, d_model: Int, d_ff: Int, n_heads: Int):
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = FFN(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def __call__(mut self, x: Tensor) raises -> Tensor:
        var h = x + self.attn(self.norm1(x))
        return h + self.ffn(self.norm2(h))

    def parameters(mut self) -> List[ModuleParam]:
        var ps = self.attn.parameters()
        ps += self.ffn.parameters()
        ps += self.norm1.parameters()
        ps += self.norm2.parameters()
        return ps^


struct MultiHeadAttention(Module, Movable):
    var d_model: Int
    var n_heads: Int
    var d_head: Int
    var W_Q: nn.Linear
    var W_K: nn.Linear
    var W_V: nn.Linear
    var W_O: nn.Linear

    def __init__(out self, d_model: Int, n_heads: Int):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def __call__(mut self, x: Tensor) raises -> Tensor:
        var B = x.shape(0)
        var T = x.shape(1)

        # (B, T, D) → (B, H, T, Dh)
        var Q = self.W_Q(x).reshape((B, T, self.n_heads, self.d_head)).transpose(1, 2)
        var K = self.W_K(x).reshape((B, T, self.n_heads, self.d_head)).transpose(1, 2)
        var V = self.W_V(x).reshape((B, T, self.n_heads, self.d_head)).transpose(1, 2)

        # (B, H, T, T)
        var scores = Q.matmul(K.transpose(-2, -1))
        scores = scores / Tensor.full_like(scores, Float32(self.d_head)).sqrt()
        var mask = Tensor.full_like(scores, neg_inf[DType.float32]()).triu(1)
        var weights = (scores + mask).softmax()

        # (B, H, T, Dh) → (B, T, D)
        var ctx = weights.matmul(V).transpose(1, 2).reshape((B, T, self.d_model))
        return self.W_O(ctx)

    def parameters(mut self) -> List[ModuleParam]:
        var ps = self.W_Q.parameters()
        ps += self.W_K.parameters()
        ps += self.W_V.parameters()
        ps += self.W_O.parameters()
        return ps^


struct FFN(Module, Movable):
    var W1: nn.Linear
    var W2: nn.Linear

    def __init__(out self, d_model: Int, d_ff: Int):
        self.W1 = nn.Linear(d_model, d_ff)
        self.W2 = nn.Linear(d_ff, d_model, bias=False)

    def __call__(mut self, x: Tensor) raises -> Tensor:
        return self.W2(self.W1(x).relu())

    def parameters(mut self) -> List[ModuleParam]:
        var ps = self.W1.parameters()
        ps += self.W2.parameters()
        return ps^


def sample(probs: List[Float32]) -> Int:
    var r = Float32(random_float64())
    var cumsum = Float32(0.0)
    for i in range(len(probs)):
        cumsum += probs[i]
        if r < cumsum:
            return i
    return len(probs) - 1
