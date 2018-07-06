#!/bin/bash

if [ "$#" -ne 2 ]; then
  echo "Usage : $(basename $0) path/to/image [mode]"
  echo "        [mode] can be 0,1,2,3 or unsorted combinations"
  exit 1
fi

echo "Processing : $1"
base="$1"
base=${base%.*}

# phase 0
if [[ "$2" = *"0"* ]]; then
  echo "Phase_0..."
  matlab -nodisplay -nosplash -nojvm -r "imgpath='$1'; phase_0_extract_features; exit"
  echo "DONE!"
fi

# phase 1
if [[ "$2" = *"1"* ]]; then
  echo "Phase_1..."
  for model in ../model/*.json; do
    modelbase=${model%.*}
    python3 phase_1_patch_predict.py $base.phase_0.mat $modelbase
  done
  echo "DONE!"
fi

# phase 2
if [[ "$2" = *"2"* ]]; then
  echo "Phase_2..."
  for phase1 in $base*.phase_1.mat;do
    basephase1=${phase1%%.*}
    matlab -nodisplay -nosplash -nojvm -r "base0='$base'; base1='$basephase1'; phase_2_evaluate; quit"
  done
  echo "DONE!"
fi

# phase 3
if [[ "$2" = *"3"* ]]; then
  echo "Composing csv..."
  echo "model;count" > ${base}_compare.csv
  for json in $base*.json;do
    model=${json%%.*}
    model=${model##*_}
    echo -n $model";"
    cat $json | grep -oP "(?<=\"count\" : ).+(?=}],)"
  done >> ${base}_compare.csv
  echo "DONE!"
fi
