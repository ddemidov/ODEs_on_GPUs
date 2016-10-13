#!/bin/bash

ntry=5

for((k=1; k<=${ntry}; k+=1)); do
    echo "Run #${k}"
    for((n=256; n<=4194304; n*=2)); do
        for t in lorenz_cpu_v1 lorenz_cpu_v2; do
            echo "${t} for n=${n}"
            ./${t} ${n}
        done
    done
done
