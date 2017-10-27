#!/bin/bash

# try different Mu with data/p1/test.txt concatenated with data/p2/unlabeled_20news.txt
for mu in 0.0 0.1 0.2 0.3
do
	java -cp jars/jama.jar:bin/ HMM data/p1/train.txt data/p1/test.txt results/p2/prediction results/p2/training_log $mu &
done

wait

for mu in 0.4 0.5 0.6 0.7
do
	java -cp jars/jama.jar:bin/ HMM data/p1/train.txt data/p1/test.txt results/p2/prediction results/p2/training_log $mu &
done

wait

for mu in 0.8 0.9 1.0
do
	java -cp jars/jama.jar:bin/ HMM data/p1/train.txt data/p1/test.txt results/p2/prediction results/p2/training_log $mu &
done

wait
