#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2021 Nagoya University (Wen-Chin Huang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Script to read and find best epoch

import argparse
from io import open
import logging
import os
import sys

from pathlib import Path

def get_parser():
    parser = argparse.ArgumentParser(
        description="Extract results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--verbose", "-V", default=1, type=int, help="Verbose option")
    parser.add_argument("--upstream", type=str, required=True, help="upstream")
    parser.add_argument("--expdir", type=str, default="../../result/downstream", help="expdir")
    parser.add_argument("--start_epoch", default=4000, type=int)
    parser.add_argument("--end_epoch", default=10000, type=int)
    parser.add_argument("--step_epoch", default=1000, type=int)
    parser.add_argument(
        "--out",
        "-O",
        type=str,
        help="The output filename. " "If omitted, then output to sys.stdout",
    )
    return parser


def grep(filepath, query):
    lines = []
    with open(filepath, "r") as f:
        for line in f:
            if query in line:
                lines.append(line.rstrip())
    return lines


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    # logging info
    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    if args.verbose > 0:
        logging.basicConfig(level=logging.INFO, format=logfmt)
    else:
        logging.basicConfig(level=logging.WARN, format=logfmt)

    srcspks = ["SEF1", "SEF2", "SEM1", "SEM2"]
    task1_trgspks = ["TEF1", "TEF2", "TEM1", "TEM2"]
    task2_trgspks = ["TFF1", "TFM1", "TGF1", "TGM1", "TMF1", "TMM1"]
    epochs = list(range(args.start_epoch, args.end_epoch+args.step_epoch, args.step_epoch))

    bests = []
    for trgspk in task1_trgspks:

        #"""
        #choose by MCD
        scores = []
        for ep in epochs:
            log_file = os.path.join(args.expdir, f"a2o_vc_vcc2020_ar_{trgspk}_{args.upstream}", str(ep), "pwg_wav", "obj.log")
            result = grep(log_file, "Mean")[0].split("Mean MCD, f0RMSE, f0CORR, DDUR, CER, WER, accept rate: ")[1].split(" ")
            scores.append(result)
        best = min(scores, key=lambda x: float(x[0]))
        bests.append(best)
        #"""
           
        """Choose 10000 ep"""
        #log_file = os.path.join(args.expdir, f"a2o_vc_vcc2020_ar_{trgspk}_{args.upstream}", "10000", "pwg_wav", "obj.log")
        #result = grep(log_file, "Mean")[0].split("Mean MCD, f0RMSE, f0CORR, DDUR, CER, WER, accept rate: ")[1].split(" ")
        #bests.append(result)


    avg = [f"{(sum([float(best[i]) for best in bests]) / 4.0):.2f}" for i in range(7)]
    print(" ".join(avg))
