#!/bin/bash
#SBATCH -c 20        # Number of cores (-c)
#SBATCH -t 1-00:10     # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p seas_compute,shared,seas_gpu,tambe,serial_requeue  # Partition to submit to
#SBATCH --mem-per-cpu=32000 #8000
#SBATCH --gpus-per-node=1
#SBATCH -J $2_job_$1 #job_name
#SBATCH -o ./output/myoutput_gbt_%j.out # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e ./output/myerrors_gbt_%j.err # File to which STDERR will be written, %j inserts jobid

echo $1
#module load python/3.8.5-fasrc01 cuda/11.1.0-fasrc01 && source activate pt1.8_cuda111
module load python/3.10.9-fasrc01 cuda/11.3.1-fasrc01 #&& source activate pt1.8_cuda111
export LD_LIBRARY_PATH=/n/home05/sjohnsonyu/.conda/envs/rlhf:$LD_LIBRARY_PATH
source activate rlhf

python src/train_rlhf.py # Your file 
