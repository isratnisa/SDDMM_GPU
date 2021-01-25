CXX=g++
NVCC=nvcc
NVCC_LIB_PATH=/usr/lib/x86_64-linux-gnu
#CXXFLAGS=-fopenmp -static -O3
#CXXFLAGS=-fopenmp -fPIC -pipe  -O3
CXXFLAGS=-O3 -std=c++11
 # -g
#NVCCFLAGS +=-O3 -fopenmp -w -restrict #--ptxas-options=-v
#NVCCFLAGS +=-O3 -w -gencode arch=compute_35,code=sm_35 -rdc=true -Xptxas -dlcm=ca -Xcompiler -fopenmp
# -dc -G --restrict -m64 -Xcompiler -Wall
NVCCFLAGS += -O3 -w -arch=sm_60 -Xptxas -dlcm=ca -Xcompiler -fopenmp --std=c++11 -m64 
# -lineinfo -g -G
# NVCCFLAGS += -O3 -w -arch=sm_37 -rdc=true -Xptxas -dlcm=ca -Xcompiler -fopenmp --std=c++11 -m64 -lineinfo #-g -G
NVCCLINKFLAGS = -L$(NVCC_LIB_PATH) -lcudart
# MPILINKFLAGS = -L/usr/mpi/gcc/mvapich2-2.2a/lib/
# MPILINKFLAGS =-L/opt/mvapich2/gdr/2.2/cuda8.0/gnu/lib64/
# MPILINKFLAGS=-L/opt/mvapich2/gnu/4.8/2.2-gpu/lib/
MPIIFLAGS = -I/opt/mvapich2/intel/16.0/2.2-gpu/include/
MPILINKFLAGS=-L/opt/mvapich2/intel/16.0/2.2-gpu/lib/

# MPIIFLAGS = -I/opt/mvapich2/intel/16.0/2.3rc1-gpu/include/
# MPILINKFLAGS=-L/opt/mvapich2/intel/16.0/2.3rc1-gpu/ib/

# MPILINKFLAGS=-L/opt/mvapich2/pgi/16.5/2.2-gpu/lib/
# MPIIFLAGS =-I/opt/mvapich2/pgi/16.5/2.2-gpu/include/


all: sddmm 

sddmm: sddmm.o 
	# ${NVCC} ${NVCCFLAGS} -o sddmm sddmm.cu $(NVCCLINKFLAGS)
	mpicxx -o sddmm -L/usr/local/cuda/8.0.44/lib64 -lcudart sddmm.o -lm -O3 -fopenmp --std=c++11 
	
sddmm.o: sddmm.cu util.h kernel.h
	${NVCC} ${NVCCFLAGS} ${MPIIFLAGS} ${MPILINKFLAGS} -lmpi  -c -o sddmm.o sddmm.cu 

clean:
	rm -rf sddmm *.o 
