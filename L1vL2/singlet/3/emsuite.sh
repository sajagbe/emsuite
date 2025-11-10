#!/bin/bash
#SBATCH -J emsuiteLF
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -t 47:59:00
#SBATCH --mem=16G 
#SBATCH -p qGPU48
#SBATCH --gres=gpu:1
#SBATCH -A CHEM9C4
#SBATCH -e %J.err
#SBATCH -o %J.out
#SBATCH --exclude=acidsgcn011
#--------------------------------------------------------------#


start=$(date +%s)

emsuite LF.in << EOF
y
EOF

end=$(date +%s)
runtime=$((end-start))

echo "Runtime: ${runtime} seconds"


