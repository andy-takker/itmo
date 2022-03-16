#!/bin/bash
set -o errexit
for i in ./output/*
do
  f=$(basename "$i")
  echo "$f"
  python lab.py d "$i" "./decode/decoded_$f"
  printf "\n"
done;