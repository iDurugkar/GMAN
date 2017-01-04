#!/bin/bash
mixing=(arithmetic harmonic)
#weighting=( normal log )
lam=(0 1)
runs=(2 3 4 5)
disc=(2 5)

objective=(original modified)
for r in "${runs[@]}"
do
for o in "${objective[@]}"
do
python GMAN.py --dataset cifar --num_disc 1 --num_hidden 256 --epochs 50 --lam 0. --path "cifar/1_${o}_256_${1}"
done
done

for r in "${runs[@]}"
do
#for m in "${mixing[@]}"
#do
for l in "${lam[@]}"
do
for d in "${disc[@]}"
do
#for w in "${weighting[@]}"
#do
fname=$disc"_"$l"_256_"$r
echo $fname
python GMAN.py --dataset cifar --num_disc $disc --num_hidden 256 --epochs 50 --lam $l --path cifar/$fname
done
done
#done
done
