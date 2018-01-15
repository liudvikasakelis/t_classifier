#!/bin/bash
for f in night_school/*; do
	python 10-fold_train.py $f data/transactions/train.data4.csv
done
