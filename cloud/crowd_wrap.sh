#!/bin/bash

if [[ "$1" == "" ]]
then
	echo "Usage : $(basename $0) path/to/image"
fi

echo "Processing : $1"
base="$1"
base=${base%.*}

# phase 0
echo "Phase_0..."
matlab -nodisplay -nosplash -nojvm -r "imgpath='$1'; phase_0_extract_features; exit"
echo "DONE!"

# phase 1
echo "Phase_1..."
python3 phase_1_patch_predict.py $base.phase_0.mat
echo "DONE!"

# phase 2
echo "Phase_2..."
matlab -nodisplay -nosplash -nojvm -r "base='$base'; phase_2_evaluate; exit"
echo "DONE!"
