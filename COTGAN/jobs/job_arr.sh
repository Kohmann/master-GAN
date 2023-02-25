#!/bin/bash
for i in job*.slurm; do
    echo "sbatch $i"
    sbatch $i
done