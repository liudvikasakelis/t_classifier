#!/bin/bash
source venv/bin/activate

# get ready

for train in 16436 16801
do
	for CV in 16071 16436 
	do
		command="{\"train_cutoff\": $train, \"CV_cutoff\": $CV}"
		python 10-fold_train.py data/transactions/train.data4.csv "$command"
	done
done
