#!/bin/bash
set -o errexit
for i in ./calgarycorpus/*; do
  f=$(basename "$i")
  echo "$f"
  python lab.py e "$i" "./encode/$f"
  printf "\n"
done;