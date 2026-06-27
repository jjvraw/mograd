from model import TinyTransformer

import mograd as mg
import mograd.nn as nn
from mograd.data.friendly import tiny_shakespeare

comptime VOCAB_SIZE = 65
comptime SEQ_LEN = 256
comptime D_MODEL = 384
comptime N_HEADS = 6
comptime D_FF = 1536
comptime N_LAYERS = 6
comptime LR = Float32(3e-4)
comptime N_STEPS = 10000


def main() raises:
    var device = mg.device()
    var ds = tiny_shakespeare(device)
    var model = TinyTransformer(VOCAB_SIZE, D_MODEL, D_FF, N_LAYERS, N_HEADS)
    var opt = nn.AdamW(model.parameters(), lr=LR, weight_decay=Float32(0.1))

    for step in range(N_STEPS):
        var x, y = ds.get_batch(SEQ_LEN, batch_size=32, seed=step)

        var logits = model(x)  # (B, T, VOCAB_SIZE)
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
