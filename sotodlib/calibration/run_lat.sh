#!/bin/bash

ntasks=8
indir=/home/zhileixu/data/simulated_tod/LAT/i6
outdir=/home/zhileixu/data/fitting_results

for start_day in {1..5}; do
    mpirun -n ${ntasks} ./calibration_tod_analysis.py \
            --indir ${indir} \
            --outdir ${outdir} \
            --f_format g3 \
            --tele LAT \
            --tube i6 \
            --band f090 \
            --wafer w13 \
            --sso_name Saturn \
            --year 2022 \
            --month 9 \
            --day $start_day \
            --highpass \
            --cutoff 0.2
