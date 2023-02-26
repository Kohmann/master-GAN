#!/bin/bash
for i in job[0-9].slurm; do
    echo "sbatch $i"
    sbatch $i
done