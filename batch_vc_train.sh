#!/bin/bash

upstream=$1

set -e

date +%T

pids=() # initialize pids
for trgspk in TEF1 TEF2 TEM1 TEM2; do
(
    expname=a2o_vc_vcc2020_ar_${trgspk}_${upstream}
    expdir=result/downstream/${expname}
    mkdir -p ${expdir}
    python run_downstream.py -m train \
        --config downstream/a2o-vc-vcc2020/config_ar.yaml \
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
