#!/bin/bash

ntasks=8
indir=/home/zhileixu/data/simulated_tod/SAT/SAT1
outdir=/home/zhileixu/data/fitting_results

mpirun -n ${ntasks} ./calibration_tod_analysis.py \
        --indir ${indir} \
        --outdir ${outdir} \
        --f_format g3 \
        --tele SAT \
        --tube SAT1 \
        --band f090 \
        --wafer w25 \
        --sso_name Jupiter \
        --year 2022 \
        --month 7 \
        --day 10 \
        --highpass \
        --cutoff 0.2 \
