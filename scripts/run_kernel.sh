#!/bin/bash
module load cuda/12.4.1 cudnn
num="$1"
../build/sgemm "$num"