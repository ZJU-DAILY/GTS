// Data Partition
// Created on 24-01-05

#pragma once
#include "pivot.cuh"
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include "config.cuh"

#define CHECK(call)                                                \
	{                                                              \
		const cudaError_t error = call;                            \
		if (error != cudaSuccess)                                  \
		{                                                          \
			fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
			fprintf(stderr, "code: %d, reason: %s\n", error,       \
					cudaGetErrorString(error));                    \
			exit(1);                                               \
		}                                                          \
	}

__managed__ int PNUM = 5;

// Check if the pivot id is satisfied
__global__ void check(int *accu_upper, int *flag, int num, int pid)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int total_num = gridDim.x * blockDim.x;

	for (int idx = id; idx < num; idx += total_num)
	{
		if (flag[idx] == pid)
		{
			// accu[idx] = 1;
			accu_upper[idx] = 1;
		}
		else
		{
			// accu[idx] = 0;
			accu_upper[idx] = 0;
		}
	}
}

// Compute accu array
__global__ void getAccu(int *accu, int *accu_upper, int level_curr, int num)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int total_num = gridDim.x * blockDim.x;

	for (int idx = id; idx < num; idx += total_num)
	{
		int i = idx - pow(2, level_curr - 1);

		if (i >= 0)
		{
			accu[idx] = accu_upper[idx] + accu_upper[i];
		}
		else
		{
			accu[idx] = accu_upper[idx];
		}
	}
}

// Update upper array of accu
__global__ void updateUpp(int *accu, int *accu_upper, int num)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int total_num = gridDim.x * blockDim.x;

	for (int idx = id; idx < num; idx += total_num)
	{
		accu_upper[idx] = accu[idx];
	}
}

// Get merge result according to accu
__global__ void getMerge(int *accu, Obj obj, Obj obj_m, int num, int count)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int total_num = gridDim.x * blockDim.x;

	for (int idx = id; idx < num; idx += total_num)
	{
		if (accu[idx] > 0 && idx == 0)
		{
			int i = accu[idx];
			obj_m.res_id[i - 1 + count] = obj.res_id[idx];
			obj_m.dis[i - 1 + count] = obj.dis[idx];
		}
		else if (idx > 0)
		{
			if (accu[idx] > accu[idx - 1])
			{
				int i = accu[idx];
				obj_m.res_id[i - 1 + count] = obj.res_id[idx];
				obj_m.dis[i - 1 + count] = obj.dis[idx];
			}
		}
	}
}

__global__ void showRes(float *radius)
{
	for (int i = 0; i < PNUM; i++)
	{
		printf("%f\n", radius[i]);
	}
}

// Data partition
void getPartition(int num, Obj &obj_m, Obj obj_p, int pnum, int *&part_num, float *&radius)
{
	cout << "Data partition..." << endl;

	int *accu; // accu array
	// int* accu_upper; // upper array of accu
	// int level; // level for computing accu
	int block_num = (num + THREAD_NUM - 1) / THREAD_NUM;
	int idx = (num - 1);
	int count = 0;
	Obj obj_temp;
	Obj obj_final;
	int mnum[1];

	cudaMalloc((void **)&obj_temp.dis, block_num * sizeof(float));
	cudaMalloc((void **)&obj_temp.res_id, block_num * sizeof(int));
	cudaMalloc((void **)&obj_final.dis, sizeof(float));
	cudaMalloc((void **)&obj_final.res_id, sizeof(int));
	// level = ceil(log2(num)) + 1;
	cudaMalloc((void **)&accu, (num) * sizeof(int));
	// cudaMallocManaged((void**)&accu_upper, (num) * sizeof(int));

	for (int i = 0; i < pnum; i++)
	{
		check<<<block_num, THREAD_NUM>>>(accu, obj_p.flag, num, i);
		cudaDeviceSynchronize();

		/*for (int j = 1; j < level; j++) {
			getAccu << <block_num, THREAD_NUM >> > (accu, accu_upper, j, num);
			cudaDeviceSynchronize();
			updateUpp << <block_num, THREAD_NUM >> > (accu, accu_upper, num);
			cudaDeviceSynchronize();
		}*/

		thrust::inclusive_scan(thrust::device, accu, accu + num, accu);
		cudaDeviceSynchronize();

		// int mnum = accu[idx];
		// part_num[i] = mnum;
		////cout << mnum << " ";
		// count += mnum;

		// cudaMalloc((void**)&obj_m[i].dis, (mnum[0]) * sizeof(float));
		// cudaMalloc((void**)&obj_m[i].res_id, (mnum[0]) * sizeof(int));
		////cudaMalloc((void**)&obj_m[i].flag, (mnum) * sizeof(int));

		getMerge<<<block_num, THREAD_NUM>>>(accu, obj_p, obj_m, num, count);
		cudaDeviceSynchronize();

		cudaMemcpy(mnum, accu + num - 1, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(part_num + i, accu + num - 1, sizeof(int), cudaMemcpyDeviceToDevice);
		cudaDeviceSynchronize();

		/*int start_idx = count;
		int end_idx = count + mnum[0];
		thrust::sort_by_key(thrust::device, obj_m.dis + start_idx, obj_m.dis + end_idx , obj_m.res_id + start_idx);*/

		int block_num_m = (mnum[0] + THREAD_NUM - 1) / THREAD_NUM;

		findMax<<<block_num_m, THREAD_NUM>>>(obj_m, mnum[0], obj_temp, count);
		cudaDeviceSynchronize();
		findMax<<<1, THREAD_NUM>>>(obj_temp, block_num_m, obj_final, 0);
		cudaDeviceSynchronize();

		count += mnum[0];
		cudaMemcpy((radius + i), obj_final.dis, sizeof(float), cudaMemcpyDeviceToDevice);
	}

	// cout << count << endl;

	cudaFree(obj_temp.res_id);
	cudaFree(obj_temp.dis);
	cudaFree(obj_final.res_id);
	cudaFree(obj_final.dis);
	cudaFree(accu);
	// cudaFree(accu_upper);
}

// void getPartition(int num, Obj* obj_m, Obj obj_p, int pnum) {
//
//	cout << "Data partition..." << endl;
//
//	int* accu[PNUM]; // accu array
//	int* accu_upper[PNUM]; // upper array of accu
//	int level; // level for computing accu
//	int block_num = (num + THREAD_NUM - 1) / THREAD_NUM;
//	int idx = (num - 1);
//	int mnum[PNUM];
//	int count = 0;
//	level = ceil(log2(num)) + 1;
//	cudaStream_t s_kernal[PNUM];
//	//cudaMallocManaged((void**)&accu, (num) * sizeof(int));
//	//cudaMallocManaged((void**)&accu_upper, (num) * sizeof(int));
//
//	for (int i = 0; i < pnum; i++) {
//		cudaStreamCreate(&s_kernal[i]);
//
//		//int* accu; // accu array
//		//int* accu_upper; // upper array of accu
//		cudaMallocManaged((void**)&accu[i], (num) * sizeof(int));
//		cudaMallocManaged((void**)&accu_upper[i], (num) * sizeof(int));
//
//		check << <block_num, THREAD_NUM, 0, s_kernal[i] >> > (accu_upper[i], obj_p.flag, num, i);
//		cudaStreamSynchronize(s_kernal[i]);
//
//		for (int j = 1; j < level; j++) {
//			getAccu << <block_num, THREAD_NUM, 0, s_kernal[i] >> > (accu[i], accu_upper[i], j, num);
//			cudaStreamSynchronize(s_kernal[i]);
//			updateUpp << <block_num, THREAD_NUM, 0, s_kernal[i] >> > (accu[i], accu_upper[i], num);
//			cudaStreamSynchronize(s_kernal[i]);
//		}
//
//		mnum[i] = accu[i][idx];
//		//cout << mnum[i] << endl;
//		count += mnum[i];
//
//		cudaMallocManaged((void**)&obj_m[i].dis, (mnum[i]) * sizeof(float));
//		cudaMallocManaged((void**)&obj_m[i].res_id, (mnum[i]) * sizeof(int));
//		cudaMallocManaged((void**)&obj_m[i].flag, (mnum[i]) * sizeof(int));
//
//		getMerge << <block_num, THREAD_NUM, 0, s_kernal[i] >> > (accu[i], obj_p, obj_m[i], num);
//		cudaStreamSynchronize(s_kernal[i]);
//
//		cudaFree(accu[i]);
//		cudaFree(accu_upper[i]);
//		cudaStreamDestroy(s_kernal[i]);
//	}
//
//	cout << count << endl;
// }