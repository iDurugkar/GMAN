#!/usr/bin/env bash

mixing=( arithmetic harmonic )
#weighting=( normal log )
lam=( 0. 1.)
runs=(1 2 3 4)
disc=(2 5)

objective=(original modified)
for o in "${objective[@]}"
do
python GMAN.py --dataset cifar --num_disc 1 --num_hidden 256 --epochs 50 --lam 0. --path cifar/1_
done

for r in "${runs[@]}"
do
for m in "${mixing[@]}"
do
for l in "${lam[@]}"
do
#for w in "${weighting[@]}"
#do
fname=$disc"_"$m"_"$l"_$r"
echo $fname
python multi_disc_GAN.py --dataset mnist --num_disc $disc --mixing $m --num_hidden 128 --weighting normal --epochs 20 --lam $l --path mnist/$fname
# --lam $l --weighting $w
done
done
done
