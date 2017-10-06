#!/bin/bash

java -cp jars/jama.jar:bin/ HMM data/p1/train.txt data/p1/test.txt results/p1/prediction.txt results/p1/training_log.txt
