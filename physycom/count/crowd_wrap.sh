#!/bin/bash

if [ "$#" -ne 3 ]; then
  echo "Usage : $(basename $0) path/to/image [mode] path/to/model/base"
  echo "        [mode] can be 0,1,2 or unsorted combinations"
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
  python3 phase_1_patch_predict.py $base.phase_0.mat $3
  echo "DONE!"
fi

# phase 2
if [[ "$2" = *"2"* ]]; then
  echo "Phase_2..."
  matlab -nodisplay -nosplash -nojvm -r "base0='$base'; base1='$base'; phase_2_evaluate; quit"
  echo "DONE!"
fi
