# pico-gpt

Code golf of Karpathy's [micro-gpt](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) — a complete GPT-2 training and inference implementation in pure, dependency-free Python.

## Versions

| File | Lines | Chars | Non-whitespace chars | Description |
|---|---|---|---|---|
| `microgpt.py` | 154 | 6819 | 5089 | Karpathy's original |
| `picogpt.py` | 32 | 1998 | 1800 | Golfed: `class V`, standalone backward, S-list, inlined ops |
| `picogpt_v2.py` | 84 | 1873 | 1589 | Further golfed: paired children/grads, removed `__truediv__`, flattened params |
| `picogpt_formatted.py`| 95 | 2761 | 1770 | Formatted version of `picogpt.py` for code study |
| `picogpt_v2_formatted.py`| 96 | 2578 | 1607 | Formatted version of `picogpt_v2.py` for code study |
| `picogpt_commented.py`| 540 | — | — | Heavily commented educational guide |

`picogpt_v2.py` is a 69% reduction in non-whitespace characters vs the original (1589 vs 5089). All versions produce bit-identical output (same losses, same generated samples).

## Install
```
uv venv
```

## Usage
Download `input.txt` (names dataset) on first run, trains for 1000 steps, then generates 20 names.

Download dataset
```
./download.sh
```

Run training
```
uv run picogpt_v2.py
```