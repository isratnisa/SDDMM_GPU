export CUDA_VISIBLE_DEVICES="0"
# change i to 4 if u want to run on 4 process
nvidia-smi -i 2 -c EXCLUSIVE_PROCESS
#this will throw a err but no worries
nvidia-cuda-mps-control -d
# quit from MPS
echo quit | nvidia-cuda-mps-control
nvidia-smi -i 2 -c DEFAULT
