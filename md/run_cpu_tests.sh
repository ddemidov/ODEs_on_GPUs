#!/bin/bash

ntry=5

for((k=1; k<=${ntry}; k+=1)); do
    echo "Run #${k}"
    for((n=16; n<=1024; n*=2)); do
        m=$((1024 / ${n}))
        echo "mdc_cpu_v2 for n=${n}"
        ./mdc_cpu_v2 ${n} ${m}
    done
done

for((k=1; k<=${ntry}; k+=1)); do
    echo "Run #${k}"
    for((n=16; n<=512; n*=2)); do
        m=$((512 / ${n}))
        echo "mdc_cpu_v1 for n=${n}"
        ./mdc_cpu_v1 ${n} ${m}
    done
done

echo "Single run of mdc_cpu_v1 for n=1024"
./mdc_cpu_v1 1024 1
