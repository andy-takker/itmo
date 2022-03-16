#!/bin/bash
set -o errexit
for i in ./calgarycorpus/*; do
  f=$(basename "$i")
  echo "$f"
  python lab.py e "$i" "./output/encoded_$f"
  printf "\n\n"
done;