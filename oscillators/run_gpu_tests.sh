#!/bin/bash

export OCL_DEVICE=K20

ntry=5

for((k=0; k<${ntry}; k+=1)); do
    echo "Run #${k}"
    for((n=256; n<=4194304; n*=2)); do
        for t in po_thrust po_vexcl; do
            echo "${t} for n=${n}"
            ./${t} ${n}
        done
    done
done
