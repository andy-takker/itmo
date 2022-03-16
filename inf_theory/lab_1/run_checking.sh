#!/bin/bash
set -o errexit
for i in ./calgarycorpus/*
do
  f=$(basename "$i")
  echo "$f"
  python lab.py c "$i" "./decode/decoded_encoded_$f"
  printf "\n"
done