#!/bin/bash

export OCL_DEVICE=K20

ntry=5

for((k=0; k<${ntry}; k+=1)); do
    echo "Run #${k}"
    for((n=16; n<=1024; n*=2)); do
        m=$((1024 / ${n}))
        for t in mdc_thrust_v1 mdc_thrust_v2 mdc_vexcl_v1 mdc_vexcl_v2 mdc_vexcl_v3; do
            echo "${t} for n=${n}"
            ./${t} ${n} ${m}
        done
    done
done
