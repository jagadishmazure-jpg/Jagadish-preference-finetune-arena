"""Train the mini-GPT on the GPU this process can see, and save the checkpoint.

Azure ML runs this from llm-foundations/azure/minigpt-job.yml. On a machine with no CUDA
device it still trains and saves, on CPU, so the script can be checked without a GPU.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from llmfound.minigpt import MiniGPT, prepare, save_checkpoint, train_steps  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=400)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    tok, ids = prepare(num_merges=40, repeats=8)
    config = {"n_embd": 128, "n_head": 4, "n_layer": 4, "block_size": 128, "vocab": tok.vocab_size}
    model = MiniGPT(tok.vocab_size, n_embd=128, n_head=4, n_layer=4, block_size=128)
    losses = train_steps(model, ids, steps=args.steps, batch=16, block=64, lr=3e-4, seed=0, device=device)
    save_checkpoint(Path(args.output), model, tok, {**config, "device": device})
    print(f"device={device} final_loss={losses[-1]} saved={args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
