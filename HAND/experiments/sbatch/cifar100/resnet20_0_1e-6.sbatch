#!/bin/bash

################################################################################################
### sbatch configuration parameters must start with #SBATCH and must precede any other commands.
### To ignore, just add another # - like so: ##SBATCH
################################################################################################

#SBATCH --partition rtx2080			### specify partition name where to run a job. main: all nodes; gtx1080: 1080 gpu card nodes; rtx2080: 2080 nodes; teslap100: p100 nodes; titanrtx: titan nodes
#SBATCH --time 5-00:00:00			### limit the time of job running. Make sure it is not greater than the partition time limit!! Format: D-H:MM:SS
#SBATCH --job-name resnet20_1e-6		### name of the job
#SBATCH --output resnet20_1e-6_%J.out			### output log for running job - %J for job number
#SBATCH --mail-user=maorash@post.bgu.ac.il	### user's email for sending job status messages
#SBATCH --mail-type=ALL			### conditions for sending the email. ALL,BEGIN,END,FAIL, REQUEU, NONE

#SBATCH --gpus=1				### number of GPUs, allocating more than 1 requires IT team's permission
#SBATCH --mem=32G				### ammount of RAM memory, allocating more than 60G requires IT team's permission
#SBATCH --cpus-per-task=10			### number of CPU cores, allocating more than 10G requires IT team's permission

### Print some data to output file ###
echo `date`
echo -e "\nSLURM_JOBID:\t\t" $SLURM_JOBID
echo -e "SLURM_JOB_NODELIST:\t" $SLURM_JOB_NODELIST "\n\n"

### Start your code below ####
cd /home/maorash/HANDCompress/HAND/tasks/cifar10
module load anaconda				### load anaconda module (must be present when working with conda environments)
source activate hand_pytorch
export LD_LIBRARY_PATH=/home/maorash/.conda/envs/hand_pytorch/lib/
export PYTHONPATH="${PYTHONPATH}:/home/maorash/HANDCompress/"
/home/maorash/.conda/envs/hand_pytorch/bin/python cifar10_akamaster_train.py --exp-name=resnet20_B_cifar100_1e-6 --basic-block-option B --arch resnet20 --epochs 300 --cifar_100 --cosine-smoothness-factor 1e-6