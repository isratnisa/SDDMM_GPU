CXX=g++
NVCC=nvcc
NVCC_LIB_PATH=/usr/lib/x86_64-linux-gnu
#CXXFLAGS=-fopenmp -static -O3
#CXXFLAGS=-fopenmp -fPIC -pipe  -O3
CXXFLAGS=-O3 -std=c++11 -g
#NVCCFLAGS +=-O3 -fopenmp -w -restrict #--ptxas-options=-v
#NVCCFLAGS +=-O3 -w -gencode arch=compute_35,code=sm_35 -rdc=true -Xptxas -dlcm=ca -Xcompiler -fopenmp
# -dc -G --restrict -m64 -Xcompiler -Wall
NVCCFLAGS += -O3 -w -arch=sm_37 -rdc=true -Xptxas -dlcm=ca -Xcompiler -fopenmp --std=c++11 -m64 -lineinfo #-g -G
NVCCLINKFLAGS = -L$(NVCC_LIB_PATH) -lcudart
VERSION=1.41

all: sddmm 

sddmm: sddmm.cu util.h 
	${NVCC} ${NVCCFLAGS} -o sddmm sddmm.cu $(NVCCLINKFLAGS)

# omp-pmf-predict: pmf-predict.cpp pmf.h util.o
# 	${CXX} ${CXXFLAGS} -o omp-pmf-predict pmf-predict.cpp  util.o 

ccd-r1.o: ccd-r1.cu util.o
	${NVCC} ${NVCCFLAGS} -c -o ccd-r1.o ccd-r1.cu 

util.o: util.h util.cpp
	${CXX} ${CXXFLAGS} -c -o util.o util.cpp
clean:
	rm -rf sddmm *.o 
