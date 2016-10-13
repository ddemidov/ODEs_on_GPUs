#!/bin/bash

ntry=5

for((k=1; k<=${ntry}; k+=1)); do
    echo "Run #${k}"
    for((n=256; n<=4194304; n*=2)); do
        echo "po_cpu for n=${n}"
        ./po_cpu ${n}
    done
done
