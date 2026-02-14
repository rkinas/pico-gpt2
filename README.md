# pico-gpt

Code golf of Karpathy's [micro-gpt](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) — a complete GPT-2 training and inference implementation in pure, dependency-free Python.

## Versions

| File | Lines | Chars | Description |
|---|---|---|---|
| `microgpt.py` | 192 | 8991 | Karpathy's original |
| `picogpt.py` | 8 | 2489 | Minified variable names, lambda operators, `type()` class |
| `picogpt_formatted.py`| 1 | 3915 | Formatted (ruff) version of `picogpt.py` for code study |


All versions produce bit-identical output (same losses, same generated outputs).

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
uv run picogpt.py
```