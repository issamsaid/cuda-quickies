#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>

// program parameters:
#define     N  128 // the cube size is N^3 
#define     F  4   // the filter size id F^3
#define     W  9
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
int median_filter_check(DTYPE *gpu_tab,
	                      DTYPE *cpu_tab, DTYPE *tmp, int size) {
	CUCHK(cudaMemcpy(tmp, gpu_tab, sizeof(DTYPE)*size, cudaMemcpyDeviceToHost));
	int errors = 0;
	for (int i=0; i<size; i++) if (fabs(tmp[i]-cpu_tab[i])>1.0e-4) errors++;
	return errors;
} 

// the CPU 3D filter:
int median_filter_cpu(DTYPE *tab, DTYPE *res, int ldim) {
	for (int k=F; k<N+F; k++) {
		for (int j=F; j<N+F; j++) {
			for (int i=F; i<N+F; i++) {
				int idx = i+ldim*(ldim*k+j); 
				DTYPE v = 0;
				// apply the filter on each point, average (2*F+1)^3:
				for (int lk=0; lk<2*F+1; lk++) {
					for (int lj=0; lj<2*F+1; lj++) {
						for (int li=0; li<2*F+1; li++) {
							v += tab[(i-F+li)+ldim*(ldim*(k-F+lk)+(j-F+lj))];
						}
					}
				}
				// save the result on the output tab:
				res[idx] = v/((2*F+1)*(2*F+1)*(2*F+1));
			}
		}
	}
	return 0;
}

// the GPU 3D filter kernels:
#define BLOCKZ 8
#define BLOCKY 8
#define BLOCKX 8

// the naive algorithm using global memory:
__global__ void median_filter_gpu(DTYPE *gtab, DTYPE *gres, int ldim) {
	int ix = blockIdx.x * BLOCKX + threadIdx.x + F;
	int iy = blockIdx.y * BLOCKY + threadIdx.y + F;

	off_t pln = (off_t)(ldim*ldim);
	off_t idx = (off_t)((F*pln)+ldim*iy+ix); 
	if((ix<(N+F)) && (iy<(N+F))) {
		for (int iz=F; iz<N+F; iz++) {
			DTYPE v = 0;
			// apply the filter on each point, average (2*F+1)^3:
			for (int lz=0; lz<2*F+1; lz++) {
				for (int ly=0; ly<2*F+1; ly++) {
					for (int lx=0; lx<2*F+1; lx++) {
						v += gtab[(ix-F+lx)+ldim*(ldim*(iz-F+lz)+(iy-F+ly))];
					}
				}
			}
			// save the result on the output tab:
			gres[idx] = v/((2*F+1)*(2*F+1)*(2*F+1));
			idx += pln;
		}
	}
}

__device__ void sm_read_plane(DTYPE shm[BLOCKY+2*F][BLOCKX+2*F], DTYPE *gtab, 
													    int ldim,
													    int sx, int sy, 
	                            int ix, int iy, int iz) {
	shm[sy][sx] = gtab[(ix)+ldim*(ldim*(iz)+(iy))];
	// +
	if ((sy >= BLOCKY) && (iy<N+F)) {
		shm[sy+F][sx] = gtab[(ix)+ldim*(ldim*(iz)+(iy+F))];
	}
	if ((sx >= BLOCKX) && (ix<N+F)) {
		shm[sy][sx+F] = gtab[(ix+F)+ldim*(ldim*(iz)+(iy))];
	}
	if ((sy >= BLOCKY) && (sx >= BLOCKX) && (iy<N+F) && (ix<N+F)) {
		shm[sy+F][sx+F] = gtab[(ix+F)+ldim*(ldim*(iz)+(iy+F))];			
	}
	if ((sy >= BLOCKY) && (sx < 2*F) && (iy<N+F) && (ix<N+F)) {
		shm[sy+F][sx-F] = gtab[(ix-F)+ldim*(ldim*(iz)+(iy+F))];			
	}
	// -
	if (sy < 2*F) {
		shm[sy-F][sx] = gtab[(ix)+ldim*(ldim*(iz)+(iy-F))];
	}
	if (sx < 2*F) {
		shm[sy][sx-F] = gtab[(ix-F)+ldim*(ldim*(iz)+(iy))];
	}
	if ((sx < 2*F) && (sy < 2*F)) {
		shm[sy-F][sx-F] = gtab[(ix-F)+ldim*(ldim*(iz)+(iy-F))];
	}
	if ((sx >= BLOCKX) && (sy < 2*F)) {
		shm[sy-F][sx+F] = gtab[(ix+F)+ldim*(ldim*(iz)+(iy-F))];
	}
	__syncthreads();
}

__device__ void rg_copy_plane(DTYPE rg[W][W][W],
											        DTYPE shm[BLOCKY+2*F][BLOCKX+2*F],
											        int sx, int sy, int rz) {
	for (int ly=-F; ly<=F; ly++) {
		for (int lx=-F; lx<=F; lx++) {
			rg[rz][ly+F][lx+F] = shm[sy+ly][sx+lx];
		}
	}
}

__device__ void rg_shift_plane(DTYPE rg[W][W][W]) {
	for (int lz=0; lz<W-1; lz++) {
		for (int ly=0; ly<W; ly++) {
			for (int lx=0; lx<W; lx++) {
				rg[lz][ly][lx] = rg[lz+1][ly][lx];
			}
		}
	}
}

// this version uses shared memory and registers:
__global__ void median_filter_gpu_sm_2d(DTYPE *gtab, DTYPE *gres, int ldim) {
	__shared__ DTYPE shm[BLOCKY+2*F][BLOCKX+2*F];
	int sx = threadIdx.x+F;
	int sy = threadIdx.y+F;

	int ix = blockIdx.x * BLOCKX + sx;
	int iy = blockIdx.y * BLOCKY + sy;

	off_t pln = (off_t)ldim*ldim;
	off_t idx = (off_t)(F*pln)+ldim*iy+ix; 

	DTYPE rg[W][W][W];
	// bootstrap the registers:
	for (int rz=0; rz<(2*F); rz++) {
		sm_read_plane(shm, gtab, ldim, sx, sy, ix, iy, rz);
		rg_copy_plane(rg, shm, sx, sy, rz);
	}

	for (int iz=F; iz<N+F; iz++) {
		//if((ix<(N+F)) && (iy<(N+F)) && (iz<(N+F))) {
			sm_read_plane(shm, gtab, ldim, sx, sy, ix, iy, iz+F);
			rg_copy_plane(rg, shm, sx, sy, 2*F);
			DTYPE v = 0;
			for (int lz=0; lz<W; lz++) {	
				for (int ly=0; ly<W; ly++) {
					for (int lx=0; lx<W; lx++) {
						v += rg[lz][ly][lx];
					}
				}
			}
			rg_shift_plane(rg);
			/*
			// apply the filter on each point, average (2*F+1)^3:
			for (int lz=0; lz<F; lz++) {
				for (int ly=0; ly<2*F+1; ly++) {
					for (int lx=0; lx<2*F+1; lx++) {
						//v += shm[threadIdx.z+lz][threadIdx.y+ly][threadIdx.x+lx];
						v += gtab[(ix-F+lx)+ldim*(ldim*(iz-F+lz)+(iy-F+ly))];
					}
				}
			}

				for (int ly=-F; ly<=F; ly++) {
					for (int lx=-F; lx<=F; lx++) {
						v += shm[sy+ly][sx+lx];
					}
				}
		
			for (int lz=F+1; lz<2*F+1; lz++) {
				for (int ly=0; ly<2*F+1; ly++) {
					for (int lx=0; lx<2*F+1; lx++) {
						//v += shm[threadIdx.z+lz][threadIdx.y+ly][threadIdx.x+lx];
						v += gtab[(ix-F+lx)+ldim*(ldim*(iz-F+lz)+(iy-F+ly))];
					}
				}
			}
			*/
			// save the result on the output tab:
			gres[idx] = v/((2*F+1)*(2*F+1)*(2*F+1));
			idx      += pln;
		//}
	}
}

// this version uses shared memory:
__global__ void median_filter_gpu_sm_3d(DTYPE *gtab, DTYPE *gres, int ldim) {
	__shared__ DTYPE shm[BLOCKZ+2*F][BLOCKY+2*F][BLOCKX+2*F];
	int sx = threadIdx.x + F;
	int sy = threadIdx.y + F;
	int sz = threadIdx.z + F;

	int ix = blockIdx.x * BLOCKX + sx;
	int iy = blockIdx.y * BLOCKY + sy;
	
	off_t pln = (off_t)ldim*ldim;
	off_t idx = (off_t)(sz*pln)+ldim*iy+ix; 
	
	for (int iz=threadIdx.z+F; iz<N+F; iz+=BLOCKZ) {
		
		if((ix<(N+F)) && (iy<(N+F)) && (iz<(N+F))) {
			// the main entry:
			shm[sz][sy][sx] = gtab[(ix)+ldim*(ldim*(iz)+(iy))];
			// front:
			if ((sz < 2*F) && (sy < 2*F) && (sx < 2*F)) {
				shm[sz-F][sy-F][sx-F] = gtab[(ix-F)+ldim*(ldim*(iz-F)+(iy-F))];
			}
			if ((sz < 2*F) && (sy < 2*F)) {
				shm[sz-F][sy-F][sx] = gtab[(ix)+ldim*(ldim*(iz-F)+(iy-F))];
			}
			if ((sz < 2*F) && (sy < 2*F) && (sx >= BLOCKX)) {
				shm[sz-F][sy-F][sx+F] = gtab[(ix+F)+ldim*(ldim*(iz-F)+(iy-F))];
			}
			if ((sz < 2*F) && (sx < 2*F)) {
				shm[sz-F][sy][sx-F] = gtab[(ix-F)+ldim*(ldim*(iz-F)+(iy))];
			}
			if (sz < 2*F) {
				shm[sz-F][sy][sx] = gtab[(ix)+ldim*(ldim*(iz-F)+(iy))];
			}
			if ((sz < 2*F) && (sx >= BLOCKX)) {
				shm[sz-F][sy][sx+F] = gtab[(ix+F)+ldim*(ldim*(iz-F)+(iy))];			
			}
      if ((sz < 2*F) && (sy >= BLOCKY) && (sx < 2*F)) {
				shm[sz-F][sy+F][sx-F] = gtab[(ix-F)+ldim*(ldim*(iz-F)+(iy+F))];
			}			
			if ((sz < 2*F) && (sy >= BLOCKY)) {
				shm[sz-F][sy+F][sx] = gtab[(ix)+ldim*(ldim*(iz-F)+(iy+F))];			
			}
		  if ((sz < 2*F) && (sy >= BLOCKY) && (sx >= BLOCKX)) {
				shm[sz-F][sy+F][sx+F] = gtab[(ix+F)+ldim*(ldim*(iz-F)+(iy+F))];
			}
			// back:
			if ((sz >= BLOCKZ) && (sy < 2*F) && (sx < 2*F)) {
				shm[sz+F][sy-F][sx-F] = gtab[(ix-F)+ldim*(ldim*(iz+F)+(iy-F))];			
			}
			if ((sz >= BLOCKZ) && (sy < 2*F)) {
				shm[sz+F][sy-F][sx] = gtab[(ix)+ldim*(ldim*(iz+F)+(iy-F))];
			}
			if ((sz >= BLOCKZ) && (sy < 2*F) && (sx >= BLOCKX)) {
				shm[sz+F][sy-F][sx+F] = gtab[(ix+F)+ldim*(ldim*(iz+F)+(iy-F))];			
			}
			if ((sz >= BLOCKZ) && (sx < 2*F)) {
				shm[sz+F][sy][sx-F] = gtab[(ix-F)+ldim*(ldim*(iz+F)+(iy))];
			}
			if (sz >= BLOCKZ) {
				shm[sz+F][sy][sx] = gtab[(ix)+ldim*(ldim*(iz+F)+(iy))];			
			}
			if ((sz >= BLOCKZ) && (sx >= BLOCKX)) {
				shm[sz+F][sy][sx+F] = gtab[(ix+F)+ldim*(ldim*(iz+F)+(iy))];			
			}
			if ((sz >= BLOCKZ) && (sy >= BLOCKY) && (sx < 2*F)) {
				shm[sz+F][sy+F][sx-F] = gtab[(ix-F)+ldim*(ldim*(iz+F)+(iy+F))];			
			}
			if ((sz >= BLOCKZ) && (sy >= BLOCKY)) {
				shm[sz+F][sy+F][sx] = gtab[(ix)+ldim*(ldim*(iz+F)+(iy+F))];			
			}
			if ((sz >= BLOCKZ) && (sy >= BLOCKY) && (sx >= BLOCKX)) {
				shm[sz+F][sy+F][sx+F] = gtab[(ix+F)+ldim*(ldim*(iz+F)+(iy+F))];			
			}
			// right:
			if ((sy < 2*F) && (sx >= BLOCKX)) {
				shm[sz][sy-F][sx+F] = gtab[(ix+F)+ldim*(ldim*(iz)+(iy-F))];			
			}
			if (sx >= BLOCKX) {
				shm[sz][sy][sx+F] = gtab[(ix+F)+ldim*(ldim*(iz)+(iy))];
			}	
			if ((sy >= BLOCKY) && (sx >= BLOCKX)) {
				shm[sz][sy+F][sx+F] = gtab[(ix+F)+ldim*(ldim*(iz)+(iy+F))];			
			}
			// left:
			if ((sy < 2*F) && (sx < 2*F)) {
				shm[sz][sy-F][sx-F] = gtab[(ix-F)+ldim*(ldim*(iz)+(iy-F))];
			}
			if (sx < 2*F) {
				shm[sz][sy][sx-F] = gtab[(ix-F)+ldim*(ldim*(iz)+(iy))];
			}
			if ((sy >= BLOCKY) && (sx < 2*F)) {
				shm[sz][sy+F][sx-F] = gtab[(ix-F)+ldim*(ldim*(iz)+(iy+F))];
			}
			// top:
			if (sy < 2*F) {
				shm[sz][sy-F][sx] = gtab[(ix)+ldim*(ldim*(iz)+(iy-F))];
			}
			// bottom:
			if (sy >= BLOCKY) {
				shm[sz][sy+F][sx] = gtab[(ix)+ldim*(ldim*(iz)+(iy+F))];
			}
			__syncthreads();

			DTYPE v = 0;
			// apply the filter on each point, average (2*F+1)^3:
			for (int lz=-F; lz<=F; lz++) {	
				for (int ly=-F; ly<=F; ly++) {
					for (int lx=-F; lx<=F; lx++) {
						v += shm[sz+lz][sy+ly][sx+lx];
					}
				}
			}
			// save the result on the output tab:
			gres[idx] = v/((2*F+1)*(2*F+1)*(2*F+1));
			idx      += pln*BLOCKZ;
		}
	}
}

// the main program:
int main() {
	// for timing:
	clock_t _t;
  double time_used = 0.0;
	// create a 3D cube and the working array:
	int errors;
	int ldim =  2*F+N;
	int size = (2*F+N)*(2*F+N)*(2*F+N);
	DTYPE * tab = (DTYPE*)malloc(sizeof(DTYPE)*size);
	DTYPE * res = (DTYPE*)malloc(sizeof(DTYPE)*size);
	// create the arrays on the GPU:
	DTYPE *gtab=NULL, *gres = NULL;
	CUCHK(cudaMalloc((void**)&gtab, sizeof(DTYPE)*size));
	CUCHK(cudaMalloc((void**)&gres, sizeof(DTYPE)*size));
	// populate the cube with random values:
	srand(time(NULL));
	for (int i=0; i<size; i++) tab[i] = (DTYPE)(rand()%99);
	// copy to the GPU:
	CUCHK(cudaMemcpy((void*)gtab, tab,
		               sizeof(DTYPE)*size,
		               cudaMemcpyHostToDevice));
	CUCHK(cudaMemset((void*)gres, 0, sizeof(DTYPE)*size));
	// apply the 3D filter on the cube (CPU):
	_t = clock();
	median_filter_cpu(tab, res, ldim);
	time_used = (clock() - _t); ///CLOCKS_PER_SEC;
	// show timing:
	fprintf(stdout, "median_filter_cpu time      : %10.2f clocks\n", time_used);
	// apply the 3D filter on the cube (GPU):
	dim3 threads(BLOCKX, BLOCKY, 1);
	dim3 blocks((N+BLOCKX-1)/BLOCKX, (N+BLOCKY-1)/BLOCKY, 1);
	_t = clock();
	median_filter_gpu<<<blocks,threads>>>(gtab, gres, ldim);
	CUCHK(cudaDeviceSynchronize());
	time_used = (clock() - _t); ///CLOCKS_PER_SEC;
	// show timing:
	fprintf(stdout, "median_filter_gpu time      : %10.2f clocks ", time_used);
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
	median_filter_gpu_sm_2d<<<blocks_2d, threads_2d>>>(gtab, gres, ldim);
	CUCHK(cudaDeviceSynchronize());
	time_used = (clock() - _t); ///CLOCKS_PER_SEC;
	// show timing:
	fprintf(stdout, "median_filter_gpu_sm_2d time: %10.2f clocks ", time_used);
	// check results:
	if ((errors = median_filter_check(gres, res, tab, size)))
		fprintf(stdout,
						"\33[1;31mFAILURE(rate: %5.2f%%)\33[m\n", 
						100*((double)errors/(double)size));
	else fprintf(stdout, "\33[1;32mSUCCESS\33[m\n");
	// apply the 3D filter on the cube (GPU/3D_sm):
	CUCHK(cudaMemset((void*)gres, 0, sizeof(DTYPE)*size));
	dim3 threads_3d(BLOCKX, BLOCKY, BLOCKZ);
	dim3  blocks_3d((N+BLOCKX-1)/BLOCKX, (N+BLOCKY-1)/BLOCKY, 1);
	_t = clock();
	median_filter_gpu_sm_3d<<<blocks_3d, threads_3d>>>(gtab, gres, ldim);
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
	// free the memory:
	free(tab);
	free(res);
	CUCHK(cudaFree(gtab));
	CUCHK(cudaFree(gres));
	// exit:
	return EXIT_SUCCESS;
}