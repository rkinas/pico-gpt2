# nano-nano-nano-gpt2

Code golf of Karpathy's [nano-nano-gpt](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) — a complete GPT-2 training and inference implementation in pure, dependency-free Python.

## Versions

| File | Lines | Chars | Description |
|---|---|---|---|
| `train_original.py` | 192 | 8991 | Karpathy's original |
| `train_mini.py` | 7 | 3143 | Minified variable names, lambda operators, `type()` class |
| `train_tiny.py` | 1 | 2043 | zlib + base85 compressed payload |

All versions produce bit-identical output (same losses, same generated names).

## Usage

```
python3 train_tiny.py
```

Downloads `input.txt` (names dataset) on first run, trains for 1000 steps, then generates 20 hallucinated names.
