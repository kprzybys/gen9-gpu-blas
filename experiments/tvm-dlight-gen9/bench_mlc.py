"""Prefill and decode speed of one MLC-LLM model library (OpenCL), single request.

Usage: python3 bench_mlc.py <model_dir> <model_lib.so> [--prompt-tokens 512] [--gen 128] [--reps 3]

Prompt: a fixed text with a different first sentence per run (MLC's prefix cache would otherwise skip
the prefill of a repeated prompt), cut to exactly --prompt-tokens tokens. Greedy decoding, EOS ignored, so every run
generates --gen tokens. Prefill = time to first streamed token (includes tokenization, small);
decode = (gen - 1) tokens over the time between the first and the last token. One warm-up run first.
Also prints the engine's own metrics (usage.extra) of the last run when available, and the first
40 generated tokens' text, to compare outputs between libraries.
"""

import argparse
import json
import statistics
import time

from mlc_llm import MLCEngine
from mlc_llm.tokenizers import Tokenizer

TEXT = (
    "The history of the printing press begins in the fifteenth century, when movable metal type "
    "made it possible to reproduce books faster and more cheaply than ever before. Scholars, "
    "merchants and clergy quickly adopted the new technology, and within a few decades printing "
    "shops had opened in most large European cities. "
)


def make_prompt(tok, n_tokens: int, salt: str) -> str:
    # a distinct first sentence per run defeats MLC's prefix cache (it matches from the start)
    text = f"Document {salt}. " + TEXT
    while len(tok.encode(text)) < n_tokens + 8:
        text += TEXT
    ids = list(tok.encode(text))[:n_tokens]
    return tok.decode(ids)


def one_run(engine, model_dir, prompt, gen):
    t0 = time.perf_counter()
    t_first = None
    t_last = None
    n = 0
    pieces = []
    extra = None
    for chunk in engine.completions.create(
        model=model_dir,
        prompt=prompt,
        max_tokens=gen,
        temperature=0.0,
        stream=True,
        extra_body={"debug_config": {"ignore_eos": True}},
    ):
        now = time.perf_counter()
        if chunk.choices and chunk.choices[0].text:
            if t_first is None:
                t_first = now
            t_last = now
            n += 1
            pieces.append(chunk.choices[0].text)
        if getattr(chunk, "usage", None) is not None:
            extra = getattr(chunk.usage, "extra", None)
    return t_first - t0, (t_last - t_first), n, "".join(pieces), extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("model_lib")
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()

    tok = Tokenizer(a.model_dir)
    engine = MLCEngine(
        a.model_dir,
        model_lib=a.model_lib,
        device="opencl",
        mode="interactive",
        engine_config=None,
    )
    try:
        one_run(engine, a.model_dir, make_prompt(tok, a.prompt_tokens, "warm-up"), a.gen)
        pre, dec, texts = [], [], []
        extra = None
        for r in range(a.reps):
            prompt = make_prompt(tok, a.prompt_tokens, f"number {r + 1}")
            tp, td, n, text, extra = one_run(engine, a.model_dir, prompt, a.gen)
            pre.append(a.prompt_tokens / tp)
            dec.append((n - 1) / td if td > 0 else float("nan"))
            texts.append(text)
        print(f"lib {a.model_lib}")
        print(f"prompt {a.prompt_tokens} tokens, gen {a.gen}, reps {a.reps}")
        print("prefill t/s: " + " ".join(f"{x:.1f}" for x in pre) + f"  median {statistics.median(pre):.1f}")
        print("decode  t/s: " + " ".join(f"{x:.2f}" for x in dec) + f"  median {statistics.median(dec):.2f}")
        print("text[:200]:", json.dumps(texts[-1][:200]))
        if extra:
            print("engine metrics:", json.dumps(extra)[:600])
    finally:
        engine.terminate()


if __name__ == "__main__":
    main()
