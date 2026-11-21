#!/bin/zsh
cd -- "${0:A:h}"
python3 interactive.py
printf '\n終了しました。Enterで閉じます。'
read
