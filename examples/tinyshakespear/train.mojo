from std.math import cos, pi

import mograd as mg
import mograd.nn as nn
from mograd.data.friendly import tiny_shakespeare

from model import TinyTransformer

comptime VOCAB_SIZE = 65
comptime SEQ_LEN = 256
comptime D_MODEL = 512
comptime N_HEADS = 8
comptime D_FF = 2048
comptime N_LAYERS = 6
comptime LR = Float32(3e-4)
comptime N_STEPS = 10000
comptime BATCH_SIZE = 32
comptime WARMUP_STEPS = 200


def main() raises:
    var device = mg.device()
    var ds = tiny_shakespeare(device)
    var model = TinyTransformer(VOCAB_SIZE, D_MODEL, D_FF, N_LAYERS, N_HEADS)
    var opt = nn.AdamW(model.parameters(), lr=LR, weight_decay=Float32(0.1))

    for step in range(N_STEPS):
        var lr: Float32
        if step < WARMUP_STEPS:
            lr = LR * Float32(step + 1) / Float32(WARMUP_STEPS)
        else:
            var progress = Float32(step - WARMUP_STEPS) / Float32(N_STEPS - WARMUP_STEPS)
            lr = LR * Float32(0.5) * (Float32(1.0) + cos(Float32(pi) * progress))
        opt.set_lr(lr)

        var x, y = ds.get_batch(SEQ_LEN, batch_size=BATCH_SIZE, seed=step)

        var logits = model(x)
        var loss = logits.reshape((-1, VOCAB_SIZE)).cross_entropy(
            y.reshape((-1,)).one_hot(VOCAB_SIZE).cast(DType.float32)
        )

        var grads = loss.gradient(opt.params())
        opt.step(grads)

        if step % 100 == 0:
            print("step", step, "loss:", loss.item())

        if step % 500 == 0:
            var seed_tokens = List[Int64]()
            seed_tokens.append(Int64(0))
            var generated = model.generate(device, seed_tokens, 200, SEQ_LEN)
            print("---")
            print(ds.decode(generated))
            print("---")
