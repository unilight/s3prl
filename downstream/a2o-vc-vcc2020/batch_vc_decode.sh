#!/bin/bash

upstream=$1
task=$2
tag=$3

set -e

start_ep=4000
#start_ep=10000
interval=1000
end_ep=10000

if [ ${task} == "task1" ]; then
    trgspks=("TEF1" "TEF2" "TEM1" "TEM2")
elif [ ${task} == "task2" ]; then
    trgspks=("TFF1" "TFM1" "TGF1" "TGM1" "TMF1" "TMM1")
fi

for trgspk in "${trgspks[@]}"; do
    for ep in $(seq ${start_ep} ${interval} ${end_ep}); do
        echo "Objective evaluation: Ep ${ep}; trgspk ${trgspk}"
        expname=a2o_vc_vcc2020_${tag}_${trgspk}_${upstream}
        expdir=../../result/downstream/${expname}
        ./decode.sh pwg_${task}/ ${expdir}/${ep} ${trgspk} #> /dev/null 2>&1
        # grep 'Mean' ${expdir}/${ep}/pwg_wav/obj.log
    done
done

python find_best_epoch.py --upstream ${upstream} --tag ${tag} --task ${task}
