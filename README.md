# pico-gpt

Code golf of Karpathy's [micro-gpt](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) — a complete GPT-2 training and inference implementation in pure, dependency-free Python.

## Versions

| File | Lines | Chars | Description |
|---|---|---|---|
| `microgpt.py` | 192 | 8991 | Karpathy's original |
| `nanogpt.py` | 7 | 2824 | Minified variable names, lambda operators, `type()` class |
| `picogpt.py` | 1 | 2043 | zlib + base85 compressed payload |

All versions produce bit-identical output (same losses, same generated outputs).

## Install
```
uv venv
```

## Usage

```
uv run train_tiny.py
```

Downloads `input.txt` (names dataset) on first run, trains for 1000 steps, then generates 20 names.
