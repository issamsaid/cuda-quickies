#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include <cuda_runtime.h>

// program parameters:
#define     N  320 // the cube size is N^3. 
#define     F  4   // the filter radius.
#define     W  9   // the window size.
#define DTYPE  float

// CUDA helpers:
#define CUCHK(call)                                          \
do {                                                         \
  cudaError_t status = call;                                 \
  if (status != cudaSuccess) {                               \
    fprintf(stderr, "[GPU ERR] %s @ %s:%d\n",                \
            cudaGetErrorString(status), __FILE__, __LINE__); \
    exit(EXIT_FAILURE);                                      \
  }                                                          \
} while(0);

// check the results:
int median_filter_check(DTYPE gpu_tab[][N+W-1][N+W-1],
	                      DTYPE cpu_tab[][N+W-1][N+W-1],
	                      DTYPE tmp[][N+W-1][N+W-1], int size) {
	CUCHK(cudaMemcpy(tmp, gpu_tab, sizeof(DTYPE)*size, cudaMemcpyDeviceToHost));
	int errors = 0;
	for (int k=F; k<N+F; k++)
		for (int j=F; j<N+F; j++)
			for (int i=F; i<N+F; i++)
				if (fabs(tmp[k][j][i]-cpu_tab[k][j][i])>1.0e-4) {
					fprintf(stderr, "[%3d,%3d,%3d] c: %d g: %d\n", 
									k,j,i,cpu_tab[k][j][i],tmp[k][j][i]);
					errors++;
				}
	return errors;
} 

// the CPU 3D filter:
int median_filter_cpu(DTYPE tab[][N+W-1][N+W-1], 
											DTYPE res[][N+W-1][N+W-1]) {
	#pragma omp parallel for collapse(3)
	for (int k=F; k<N+F; k++) {
		for (int j=F; j<N+F; j++) {
			for (int i=F; i<N+F; i++) {
				DTYPE v = 0;
				// apply the filter on each point, average (2*F+1)^3:
				for (int lk=0; lk<2*F+1; lk++) {
					for (int lj=0; lj<2*F+1; lj++) {
						for (int li=0; li<2*F+1; li++) {
							v += tab[k-F+lk][j-F+lj][i-F+li];
						}
					}
				}
				// save the result on the output tab:
				res[k][j][i] = v/(W*W*W);
			}
		}
	}
	return 0;
}

// the GPU 3D filter kernels:
#define BLOCKZ 8
#define BLOCKY 8
#define BLOCKX 32

// the naive algorithm using global memory:
__global__ void median_filter_gpu_gm(DTYPE gtab[][N+W-1][N+W-1], 
	                                   DTYPE gres[][N+W-1][N+W-1]) {
	int ix = blockIdx.x * BLOCKX + threadIdx.x + F;
	int iy = blockIdx.y * BLOCKY + threadIdx.y + F;

	if((ix<(N+F)) && (iy<(N+F))) {
		for (int iz=F; iz<N+F; iz++) {
			DTYPE v = 0;
			// apply the filter on each point, average (2*F+1)^3:
			for (int lz=0; lz<W; lz++) {
				for (int ly=0; ly<W; ly++) {
					for (int lx=0; lx<W; lx++) {
						v += gtab[iz-F+lz][iy-F+ly][ix-F+lx];
					}
				}
			}
			// save the result on the output tab:
			gres[iz][iy][ix] = v/(W*W*W);
		}
	}
}

__device__ void sm_read_plane(DTYPE shm[BLOCKY+W-1][BLOCKX+W-1],
	                            DTYPE gtab[][N+W-1][N+W-1], 
													    int sx, int sy, 
	                            int ix, int iy, int iz) {
	
	if (sx < 2*F) {
		shm[sy][sx-F] = gtab[iz][iy][ix-F];
		if (sy < 2*F) {
			shm[sy-F][sx-F] = gtab[iz][iy-F][ix-F];
		}
		if (sy >= BLOCKY) {
			shm[sy+F][sx-F] = gtab[iz][iy+F][ix-F];
		}
	}
	if (sy < 2*F) {
		shm[sy-F][sx] = gtab[iz][iy-F][ix];
		if (sx >= BLOCKX) {
			shm[sy-F][sx+F] = gtab[iz][iy-F][ix+F];
		}
	}
	if (sx >= BLOCKX) {
		shm[sy][sx+F] = gtab[iz][iy][ix+F];
	}
	if (sy >= BLOCKY) {
		shm[sy+F][sx] = gtab[iz][iy+F][ix];
		if (sx >= BLOCKX) {
			shm[sy+F][sx+F] = gtab[iz][iy+F][ix+F];
		}
	}	
	shm[sy][sx] = gtab[iz][iy][ix];		
}

__device__ void rg_comp_plane(DTYPE rg[W],
											        DTYPE shm[BLOCKY+2*F][BLOCKX+2*F],
											        int sx, int sy, int rz) {
	DTYPE v=0;
	for (int ly=-F; ly<=F; ly++) {
		for (int lx=-F; lx<=F; lx++) {
			v += shm[sy+ly][sx+lx];
		}
	}
	rg[rz] = v;
}

__device__ void rg_shift_plane(DTYPE rg[W]) {
	for (int lz=0; lz<(W-1); lz++) {
		rg[lz] = rg[lz+1];
	}
}

// this version uses 2D shared memory and registers:
__global__ void median_filter_gpu_sm_2d(DTYPE gtab[][N+W-1][N+W-1], 
																				DTYPE gres[][N+W-1][N+W-1]) {
	__shared__ DTYPE shm[BLOCKY+W-1][BLOCKX+W-1];
	int sx = threadIdx.x+F;
	int sy = threadIdx.y+F;

	int ix = blockIdx.x * BLOCKX + sx;
	int iy = blockIdx.y * BLOCKY + sy;

	DTYPE rg[W];

	if((ix<(N+F)) && (iy<(N+F))) {
		// bootstrap the registers:
		for (int rz=0; rz<(W-1); rz++) {
			__syncthreads(); 
			sm_read_plane(shm, gtab, sx, sy, ix, iy, rz);
			__syncthreads();
			rg_comp_plane(rg, shm, sx, sy, rz);
		}
		for (int iz=F; iz<N+F; iz++) {
			__syncthreads(); 
			sm_read_plane(shm, gtab, sx, sy, ix, iy, iz+F);
			__syncthreads();
			rg_comp_plane(rg, shm, sx, sy, (W-1));
			DTYPE v = 0;
			for (int lz=0; lz<W; lz++) v += rg[lz];
			gres[iz][iy][ix] = v/(W*W*W);
			rg_shift_plane(rg);
		}
	}
}

// the main program:
int main() {
	// 3d array:
	typedef DTYPE array_t[N+W-1][N+W-1];
	//int leadPad = (((N+W-1+ 31) / 32)*32) - (N+W-1);
	//printf("lead pad : %d\n", leadPad);
	// for timing:
	clock_t _t;
  double time_used = 0.0;
	// create a 3D cube and the working array:
	int errors;
	int size = (W-1+N)*(W-1+N)*(W-1+N);
	// create the arrays on the CPU:
	array_t *tab = (array_t*)malloc(sizeof(DTYPE)*size);
	array_t *res = (array_t*)malloc(sizeof(DTYPE)*size);
	// create the arrays on the GPU:
	array_t *gtab=NULL, *gres = NULL;
	CUCHK(cudaMalloc((void**)&gtab, sizeof(DTYPE)*size));
	CUCHK(cudaMalloc((void**)&gres, sizeof(DTYPE)*size));
	// populate the cube with random values:
	memset((void*)tab, 0, sizeof(DTYPE)*size);
	srand(time(NULL));
	for (int k=F; k<N+F; k++)
		for (int j=F; j<N+F; j++)
			for (int i=F; i<N+F; i++)
				tab[k][j][i] = (DTYPE)(rand()%256);
	// copy to the GPU:
	CUCHK(cudaMemcpy((void*)gtab, tab,
		               sizeof(DTYPE)*size,
		               cudaMemcpyHostToDevice));
	CUCHK(cudaMemset((void*)gres, 0, sizeof(DTYPE)*size));
	// apply the 3D filter on the cube (CPU):
	_t = clock();
	median_filter_cpu(tab, res);
	time_used = (clock() - _t)*1000/CLOCKS_PER_SEC;
	// show timing:
	fprintf(stdout, "median_filter_cpu time      : %10.2f ms\n", time_used);
	// apply the 3D filter on the cube (GPU):
	dim3 threads(BLOCKX, BLOCKY, 1);
	dim3 blocks((N+BLOCKX-1)/BLOCKX, (N+BLOCKY-1)/BLOCKY, 1);
	_t = clock();
	median_filter_gpu_gm<<<blocks,threads>>>(gtab, gres);
	CUCHK(cudaDeviceSynchronize());
	time_used = (clock() - _t)*1000/CLOCKS_PER_SEC;
	// show timing:
	fprintf(stdout, "median_filter_gpu time      : %10.2f ms ", time_used);
	// check the results:
	if ((errors = median_filter_check(gres, res, tab, size)))
		fprintf(stdout,
						"\33[1;31mFAILURE(rate: %5.2f%%)\33[m\n", 
						100*((double)errors/(double)size));
	else fprintf(stdout, "\33[1;32mSUCCESS\33[m\n");
	// apply the 3D filter on the cube (GPU/3D_sm):
	CUCHK(cudaMemset((void*)gres, 0, sizeof(DTYPE)*size));
	dim3 threads_2d(BLOCKX, BLOCKY, 1);
	dim3  blocks_2d((N+BLOCKX-1)/BLOCKX, (N+BLOCKY-1)/BLOCKY, 1);	
	_t = clock();
	median_filter_gpu_sm_2d<<<blocks_2d, threads_2d>>>(gtab, gres);
	CUCHK(cudaDeviceSynchronize());
	time_used = (clock() - _t)*1000/CLOCKS_PER_SEC;
	// show timing:
	fprintf(stdout, "median_filter_gpu_sm_2d time: %10.2f ms ", time_used);
	// check results:
	if ((errors = median_filter_check(gres, res, tab, size)))
		fprintf(stdout,
						"\33[1;31mFAILURE(rate: %5.2f%%)\33[m\n", 
						100*((double)errors/(double)size));
	else fprintf(stdout, "\33[1;32mSUCCESS\33[m\n");
	/*
	// apply the 3D filter on the cube (GPU/3D_sm):
	CUCHK(cudaMemset((void*)gres, 0, sizeof(DTYPE)*size));
	dim3 threads_3d(BLOCKX, BLOCKY, BLOCKZ);
	dim3  blocks_3d((N+(2*F)+BLOCKX-1)/BLOCKX, (N+(2*F)+BLOCKY-1)/BLOCKY, 1);
	_t = clock();
	median_filter_gpu_sm_3d<<<blocks_3d, threads_3d>>>(gtab, gres);
	CUCHK(cudaDeviceSynchronize());
	time_used = (clock() - _t); ///CLOCKS_PER_SEC;
	// show timing:
	fprintf(stdout, "median_filter_gpu_sm_3d time: %10.2f clocks ", time_used);
	// check results:
	if ((errors = median_filter_check(gres, res, tab, size)))
		fprintf(stdout,
						"\33[1;31mFAILURE(rate: %5.2f%%)\33[m\n", 
						100*((double)errors/(double)size));
	else fprintf(stdout, "\33[1;32mSUCCESS\33[m\n");
	*/
	// free the memory:
	free(tab);
	free(res);
	CUCHK(cudaFree(gtab));
	CUCHK(cudaFree(gres));
	// exit:
	return EXIT_SUCCESS;
}
