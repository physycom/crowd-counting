#! /usr/bin/env bash

for dir in model_perf/*_photo; do
	echo $dir
	img=$(find $dir -type f -name "*tiny.jpg")
	[[ $img == "" ]] && img=$(find $dir -type f -name "*.jpg")
	for i in $img; do
		./crowd_compare.sh $i 123
	done
done
