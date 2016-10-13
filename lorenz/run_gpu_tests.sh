#!/bin/bash

export OCL_DEVICE=K20

ntry=5

for((k=0; k<${ntry}; k+=1)); do
    echo "Run #${k}"
    for((n=256; n<=4194304; n*=2)); do
        for t in lorenz_thrust_v1 lorenz_vexcl_v1 lorenz_vexcl_v2 lorenz_vexcl_v3; do
            echo "${t} for n=${n}"
            ./${t} ${n}
        done
    done
done
