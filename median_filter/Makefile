all: 
	nvcc -o median_filter -O3 \
					-Xcompiler -fopenmp -Xcompiler \
					-O3 -arch=sm_60 main.cu -lcudart -lgomp
