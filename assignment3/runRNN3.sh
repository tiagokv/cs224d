#!/bin/bash

# verbose
set -x
###################
# Update items below for each train/test
###################

# training params
epochs=30
step=1e-2

# for RNN2 only, otherwise doesnt matter
middleDim=30

model="RNN3" #either RNN, RNN2, RNN3, RNTN, or DCNN

#missing 5 on purpose, already ran
#middleDimBatch=("5" "15" "25" "30" "35" "45")
middleDimBatch=("30")
wvecDim=30

for middleDim in "${middleDimBatch[@]}"
do
########################################################
# Probably a good idea to let items below here be
########################################################
if [ "$model" == "RNN2" ]; then
    outfile="models/${model}_wvecDim_${wvecDim}_middleDim_${middleDim}_step_${step}_2.bin"
else
    outfile="models/${model}_wvecDim_${wvecDim}_step_${step}_2.bin"
fi


echo $outfile


python runNNet.py --step $step --epochs $epochs --outFile $outfile \
                --middleDim $middleDim --outputDim 5 --wvecDim $wvecDim --model $model
done
