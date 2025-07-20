#!/bin/bash
#SBATCH --job-name=seminar_nic
#SBATCH --output=output.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dominic.wittner@rwth-aache.de
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --mem=1G

python src/main.py
