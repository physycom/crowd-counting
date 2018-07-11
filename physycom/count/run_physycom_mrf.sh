#! /usr/bin/env bash

for dir in model_perf/*_photo; do
	echo "---- Processing : $dir"
	phase1=$(find $dir -type f -name "*phase_1.json")
	for p in $phase1; do
		../../MRF/physycom_mrf $p
	done
done
