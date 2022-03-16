#!/bin/bash
set -o errexit
for i in ./encode/*
do
  f=$(basename "$i")
  echo "$f"
  python lab.py d "$i" "./decode/$f"
  printf "\n"
done;