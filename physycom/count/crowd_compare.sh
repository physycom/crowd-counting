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
  echo "model;count;gt;err;relative" > ${base}_compare.csv
  gtname=$base
  #gtname=${base%_*}
  echo ${gtname}_count.csv
  if [ -f ${gtname}_count.csv ]; then
    gt=$(cat  ${gtname}_count.csv | wc -l)
  else
    gt=0
  fi
  for json in ${base}_*.json; do
    [[ "$json" == *".phase_1."* ]] && continue # to skip other json produced
    model=${json%%.*}
    model=$(echo $model | grep -oP "(?<=_model_).+")
    echo -n $model";"
    count=$(cat $json | grep -oP "(?<=\"count\" : ).+(?=}],)")
    err=$(($gt-$count))
    rel=$(echo "scale=3; $err/$gt" | bc -l)
    echo "$count;$gt;$err;$rel"
  done >> ${base}_compare.csv
  echo "DONE!"
fi
