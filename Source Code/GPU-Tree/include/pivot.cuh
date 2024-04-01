// Get the pivots
// Created on 24-01-05

#pragma once
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <iostream>
#include <ctime>
#include <cstdlib>
#include "file.cuh"
#include "config.cuh"

using namespace std;
#define INF 999999
#define THREAD_NUM 512 // thread num

// Compute the distance between data and pivots (num type)
__global__ void getDistance(short *data_d, int *data_info, int *pid, int pnum, Obj obj, char *data_s, int *size_s)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int total_num = gridDim.x * blockDim.x;

	for (int idx = id; idx < data_info[1]; idx += total_num)
	{
		int p = pnum - 1;
		float result = 0;

		if (data_info[2] == 2)
		{ // L2 distance
			if (idx == pid[p])
			{
				result = 0;
			}
			else
			{
				for (int i = 0; i < data_info[0]; i++)
				{
					result += pow(data_d[idx * data_info[0] + i] - data_d[pid[p] * data_info[0] + i], 2);
				}
				result = pow(result, 0.5);
			}
		}
		else if (data_info[2] == 1)
		{ // L1 distance
			if (idx == pid[p])
			{
				result = 0;
			}
			else
			{
				for (int i = 0; i < data_info[0]; i++)
				{
					result += abs(data_d[idx * data_info[0] + i] - data_d[pid[p] * data_info[0] + i]);
				}
			}
		}
		else if (data_info[2] == 0)
		{ // Max value
			if (idx == pid[p])
			{
				result = 0;
			}
			else
			{
				float temp = 0;
				for (int i = 0; i < data_info[0]; i++)
				{
					temp = abs(data_d[idx * data_info[0] + i] - data_d[pid[p] * data_info[0] + i]);
					if (temp > result)
						result = temp;
				}
			}
		}
		else if (data_info[2] == 5)
		{
			if (idx == pid[p])
			{
				result = 0;
			}
			float sa1 = 0, sa2 = 0, sa3 = 0;
			for (int i = 0; i < data_info[0]; i++)
			{
				sa1 += data_d[idx * data_info[0] + i] * data_d[idx * data_info[0] + i];
				sa2 += data_d[pid[p] * data_info[0] + i] * data_d[pid[p] * data_info[0] + i];
				sa3 += data_d[idx * data_info[0] + i] * data_d[pid[p] * data_info[0] + i];
			}
			sa1 = pow(sa1, 0.5);
			sa2 = pow(sa2, 0.5);
			if (sa1 * sa2 == 0)
			{
				printf("Error!!!\n");
			}
			result = sa3 / (sa1 * sa2);
			if (result > 1)
			{
				result = 0.99999999999999999;
			}
			result = abs(acos(result) * 180 / 3.1415926);
		}
		else if (data_info[2] == 6)
		{
			if (idx == pid[p])
			{
				result = 0;
			}
			int n = size_s[idx];
			int m = size_s[pid[p]];
			int table[MaxC][MaxC];
			if (n == 0)
				result = m;
			if (m == 0)
				result = n;
			if (n != 0 && m != 0)
			{
				for (int i = 0; i <= n; i++)
					table[i][0] = i;
				for (int j = 0; j <= m; j++)
					table[0][j] = j;
				for (int i = 1; i <= n; i++)
				{
					for (int j = 1; j <= m; j++)
					{
						int cost = (data_s[idx * MaxC + i - 1] == data_s[pid[p] * MaxC + j - 1]) ? 0 : 1;
						table[i][j] = 1 + min(table[i - 1][j], table[i][j - 1]);
						table[i][j] = min(table[i - 1][j - 1] + cost, table[i][j]);
					}
				}
				result = table[n][m];
			}
		}

		if (p == 0)
		{
			obj.dis[idx] = result;
			obj.res_id[idx] = idx;
			obj.flag[idx] = p;
		}
		else
		{
			if (obj.dis[idx] > result)
			{
				obj.dis[idx] = result;
				obj.flag[idx] = p;
			}
		}
	}
}

// Find max value
__global__ void findMax(Obj input, int length, Obj output, int count)
{
	__shared__ int top_per_block_id[THREAD_NUM];
	__shared__ float top_per_block_dis[THREAD_NUM];
	int top_id = 0;
	float top_dis = 0;

	for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < length; i += gridDim.x * blockDim.x)
	{
		if (input.dis[i] > top_dis)
		{
			top_id = input.res_id[i + count];
			top_dis = input.dis[i + count];
		}
	}

	top_per_block_id[threadIdx.x] = top_id;
	top_per_block_dis[threadIdx.x] = top_dis;
	__syncthreads();

	for (int len = THREAD_NUM / 2; len >= 1; len /= 2)
	{

		if (threadIdx.x < len)
		{
			if (top_per_block_dis[threadIdx.x] < top_per_block_dis[threadIdx.x + len])
			{
				top_per_block_dis[threadIdx.x] = top_per_block_dis[threadIdx.x + len];
				top_per_block_id[threadIdx.x] = top_per_block_id[threadIdx.x + len];
			}
		}
		__syncthreads();
	}

	if (blockDim.x * blockIdx.x < length)
	{
		if (threadIdx.x == 0)
		{
			output.dis[blockIdx.x] = top_per_block_dis[0];
			output.res_id[blockIdx.x] = top_per_block_id[0];
		}
	}
}

// Get pivots (num type)
void getPivot(short *data_d, int *data_info, int *pid, int pnum, Obj &obj_p, char *data_s, int *size_s)
{
	cout << "Getting pivots..." << endl;

	int block_num = (data_info[1] + THREAD_NUM - 1) / THREAD_NUM;
	// Obj obj_source;
	Obj obj_temp;
	Obj obj_final;

	cudaMalloc((void **)&obj_temp.dis, block_num * sizeof(float));
	cudaMalloc((void **)&obj_temp.res_id, block_num * sizeof(int));
	cudaMallocManaged((void **)&obj_final.dis, sizeof(float));
	cudaMallocManaged((void **)&obj_final.res_id, sizeof(int));

	srand(time(nullptr));
	// int random = rand() % data_info[1];
	// pid[0] = random;
	pid[0] = 0;

	for (int i = 1; i <= pnum; i++)
	{
		getDistance<<<block_num, THREAD_NUM>>>(data_d, data_info, pid, i, obj_p, data_s, size_s);
		cudaDeviceSynchronize();

		if (i < pnum)
		{
			findMax<<<block_num, THREAD_NUM>>>(obj_p, data_info[1], obj_temp, 0);
			cudaDeviceSynchronize();
			findMax<<<1, THREAD_NUM>>>(obj_temp, block_num, obj_final, 0);
			cudaDeviceSynchronize();

			pid[i] = obj_final.res_id[0];
		}
	}

	cudaFree(obj_temp.res_id);
	cudaFree(obj_temp.dis);
	cudaFree(obj_final.res_id);
	cudaFree(obj_final.dis);
}