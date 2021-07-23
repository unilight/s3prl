#!/bin/bash

upstream=$1

set -e

start_ep=4000
#start_ep=10000
interval=1000
end_ep=10000

for trgspk in TEF1 TEF2 TEM1 TEM2; do
    for ep in $(seq ${start_ep} ${interval} ${end_ep}); do
        echo "Objective evaluation: Ep ${ep}; trgspk ${trgspk}"
        expname=a2o_vc_vcc2020_ar_${trgspk}_${upstream}
        expdir=../../result/downstream/${expname}
        ./decode.sh pwg_task1/ ${expdir}/${ep} ${trgspk} #> /dev/null 2>&1
        # grep 'Mean' ${expdir}/${ep}/pwg_wav/obj.log
    done
done

python find_best_epoch.py --upstream ${upstream}
