#!/bin/bash

################################################################################################
### sbatch configuration parameters must start with #SBATCH and must precede any other commands.
### To ignore, just add another # - like so: ##SBATCH
################################################################################################

#SBATCH --partition main			### specify partition name where to run a job. main: all nodes; gtx1080: 1080 gpu card nodes; rtx2080: 2080 nodes; teslap100: p100 nodes; titanrtx: titan nodes
#SBATCH --time 5-00:00:00			### limit the time of job running. Make sure it is not greater than the partition time limit!! Format: D-H:MM:SS
#SBATCH --job-name hand		### name of the job
#SBATCH --output hand_%J.out			### output log for running job - %J for job number
##SBATCH --mail-user=maorash@post.bgu.ac.il	### user's email for sending job status messages
##SBATCH --mail-type=ALL			### conditions for sending the email. ALL,BEGIN,END,FAIL, REQUEU, NONE

#SBATCH --gpus=rtx_3090:1				### number of GPUs, allocating more than 1 requires IT team's permission
#SBATCH --mem=32G				### ammount of RAM memory, allocating more than 60G requires IT team's permission
#SBATCH --cpus-per-task=6			### number of CPU cores, allocating more than 10G requires IT team's permission

### Print some data to output file ###
echo `date`
echo -e "\nSLURM_JOBID:\t\t" $SLURM_JOBID
echo -e "SLURM_JOB_NODELIST:\t" $SLURM_JOB_NODELIST "\n\n"

### Start your code below ####
cd /home/maorash/HANDCompress/HAND/
module load anaconda				### load anaconda module (must be present when working with conda environments)
source activate myenv
export LD_LIBRARY_PATH=/home/maorash/.conda/envs/myenv/lib/
/home/maorash/.conda/envs/myenv/bin/python HAND_train.py --config_path experiments/resnet20/B/resnet20_1e-5_B_cifar10_larger_predictor_stochastic_ranger_lr_5e-3_nogc.yaml