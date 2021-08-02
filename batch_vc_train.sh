#!/bin/bash

upstream=$1
config=$2
tag=$3
part=$4

set -e

if [ ${part} == "first" ]; then
    trgspks=("TEF1" "TEF2")
elif [ ${part} == "last" ]; then
    trgspks=("TEM1" "TEM2")
elif [ ${part} == "all" ]; then
    trgspks=("TEF1" "TEF2" "TEM1" "TEM2")
elif [ ${part} == "fin" ]; then
    trgspks=("TFF1" "TFM1")
elif [ ${part} == "ger" ]; then
    trgspks=("TGF1" "TGM1")
elif [ ${part} == "man" ]; then
    trgspks=("TMF1" "TMM1")
fi

date +%T

pids=() # initialize pids
for trgspk in "${trgspks[@]}"; do
(
    expname=a2o_vc_vcc2020_${tag}_${trgspk}_${upstream}
    expdir=result/downstream/${expname}
    mkdir -p ${expdir}
    python run_downstream.py -m train \
        --config ${config} \
        -n ${expname} \
        -u ${upstream} \
        -d a2o-vc-vcc2020 \
        -o "config.trgspk='${trgspk}'" \
        > ${expdir}/train.log 2>&1
) &
pids+=($!) # store background pids
done
i=0; for pid in "${pids[@]}"; do wait ${pid} || ((i++)); done
[ ${i} -gt 0 ] && echo "$0: ${i} background jobs failed." && false
