#!/bin/sh
u=https://raw.githubusercontent.com/karpathy/makemore/master/names.txt
f=input.txt
[ -s "$f" ]||curl -L "$u" -o "$f"
