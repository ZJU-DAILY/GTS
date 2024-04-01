// GTS index
// Created on 24-01-05

#pragma once
#include <chrono>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <curand.h>
#include <curand_kernel.h>
#include <time.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include "file.cuh"
#include "config.cuh"

#define THREAD_NUM 512
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

__managed__ int TREE_ORDER = 10;
__managed__ int MAX_SIZE = 20;
__managed__ int MAX_H = 3;
__managed__ int DIS_CODE = 100;
__managed__ int INFI_DIS = 10000;

typedef struct TN
{
	int pid;
	float min_dis;
	int size;
	int lid;
	int is_leaf;
};

int *split_list;
int *pid_list;
double *dis_list;
int *split_num;
int cur_level;
int start_idx;

__global__ void getPivotDis(short *data_d, char *data_s, int *size_s, TN *node_list, int *split_list, double *dis_list,
							int *id_list, int start_idx, int *data_info, int *pid_list)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int nid = start_idx + bid;

	if (split_list[nid] == 1)
	{
		TN node = node_list[nid];
		int lid = node.lid;
		int size = node.size;
		int rid = lid + size - 1;
		__shared__ int pid[1];

		if (tid == 0)
		{
			// curandState_t state;
			// curand_init(nid, nid, 0, &state);
			// int random_id = lid + (rid - lid) * curand_uniform(&state);
			pid[0] = id_list[(lid + rid) / 2];
			pid_list[nid] = pid[0];
		}
		__syncthreads();

		for (int i = tid + lid; (i >= lid && i <= rid); i += THREAD_NUM)
		{
			double result = 0;
			if (data_info[2] == 2)
			{ // L2 distance
				for (int j = 0; j < data_info[0]; j++)
				{
					result += pow(data_d[id_list[i] * data_info[0] + j] - data_d[pid[0] * data_info[0] + j], 2);
				}
				result = pow(result, 0.5);
			}
			else if (data_info[2] == 1)
			{ // L1 distance
				for (int j = 0; j < data_info[0]; j++)
				{
					result += abs(data_d[id_list[i] * data_info[0] + j] - data_d[pid[0] * data_info[0] + j]);
				}
			}
			else if (data_info[2] == 0)
			{ // Max value
				float temp = 0;
				for (int j = 0; j < data_info[0]; j++)
				{
					temp = abs(data_d[id_list[i] * data_info[0] + j] - data_d[pid[0] * data_info[0] + j]);
					if (temp > result)
						result = temp;
				}
			}
			else if (data_info[2] == 5)
			{
				float sa1 = 0, sa2 = 0, sa3 = 0;
				for (int j = 0; j < data_info[0]; j++)
				{
					sa1 += data_d[id_list[i] * data_info[0] + j] * data_d[id_list[i] * data_info[0] + j];
					sa2 += data_d[pid[0] * data_info[0] + j] * data_d[pid[0] * data_info[0] + j];
					sa3 += data_d[id_list[i] * data_info[0] + j] * data_d[pid[0] * data_info[0] + j];
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
				int n = size_s[id_list[i]];
				int m = size_s[pid[0]];
				int table[M][M];
				if (n == 0)
					result = m;
				if (m == 0)
					result = n;
				if (n != 0 && m != 0)
				{
					for (int j = 0; j <= n; j++)
						table[j][0] = j;
					for (int k = 0; k <= m; k++)
						table[0][k] = k;
					for (int j = 1; j <= n; j++)
					{
						for (int k = 1; k <= m; k++)
						{
							int cost = (data_s[id_list[i] * M + j - 1] == data_s[pid[0] * M + k - 1]) ? 0 : 1;
							table[j][k] = 1 + min(table[j - 1][k], table[j][k - 1]);
							table[j][k] = min(table[j - 1][k - 1] + cost, table[j][k]);
						}
					}
					result = table[n][m];
				}
			}
			dis_list[i] = double(result / INFI_DIS + bid * DIS_CODE);
		}
	}

	else
	{
		TN node = node_list[nid];
		int lid = node.lid;
		int size = node.size;
		int rid = lid + size - 1;

		for (int i = tid + lid; (i >= lid && i <= rid); i += THREAD_NUM)
		{
			dis_list[i] = bid * DIS_CODE;
		}
	}
}

__global__ void nodeSplit(TN *node_list, int *split_list, double *dis_list, int *empty_list, int start_idx,
						  int *pid_list, short *data_d, int *id_list, int *data_info, char *data_s, int *size_s)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int nid = start_idx + bid;

	if (split_list[nid] == 1)
	{
		TN node = node_list[nid];
		int lid = node.lid;
		int size = node.size;
		int rid = lid + size - 1;
		__syncthreads();

		if (tid == 0)
			split_list[nid] = 0;

		for (int i = tid; i < TREE_ORDER; i += THREAD_NUM)
		{
			int avg_size = size / TREE_ORDER;
			TN node_child;

			node_child.lid = lid + avg_size * i;
			int id_child = nid * TREE_ORDER + i + 1;
			node_child.pid = pid_list[nid];
			if (i < TREE_ORDER - 1)
				node_child.size = avg_size;
			else
				node_child.size = size - (TREE_ORDER - 1) * avg_size;
			// node_child.min_dis = (dis_list[node_child.lid] - bid * DIS_CODE) * DIS_CODE;

			float result = 0;
			if (data_info[2] == 2)
			{ // L2 distance
				for (int j = 0; j < data_info[0]; j++)
				{
					result += pow(data_d[id_list[node_child.lid] * data_info[0] + j] - data_d[node_child.pid * data_info[0] + j], 2);
				}
				result = pow(result, 0.5);
			}
			else if (data_info[2] == 1)
			{ // L1 distance
				for (int j = 0; j < data_info[0]; j++)
				{
					result += abs(data_d[id_list[node_child.lid] * data_info[0] + j] - data_d[node_child.pid * data_info[0] + j]);
				}
			}
			else if (data_info[2] == 0)
			{ // Max value
				float temp = 0;
				for (int j = 0; j < data_info[0]; j++)
				{
					temp = abs(data_d[id_list[node_child.lid] * data_info[0] + j] - data_d[node_child.pid * data_info[0] + j]);
					if (temp > result)
						result = temp;
				}
			}
			else if (data_info[2] == 5)
			{
				float sa1 = 0, sa2 = 0, sa3 = 0;
				for (int j = 0; j < data_info[0]; j++)
				{
					sa1 += data_d[id_list[node_child.lid] * data_info[0] + j] * data_d[id_list[node_child.lid] * data_info[0] + j];
					sa2 += data_d[node_child.pid * data_info[0] + j] * data_d[node_child.pid * data_info[0] + j];
					sa3 += data_d[id_list[node_child.lid] * data_info[0] + j] * data_d[node_child.pid * data_info[0] + j];
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
				int n = size_s[id_list[node_child.lid]];
				int m = size_s[node_child.pid];
				int table[M][M];
				if (n == 0)
					result = m;
				if (m == 0)
					result = n;
				if (n != 0 && m != 0)
				{
					for (int j = 0; j <= n; j++)
						table[j][0] = j;
					for (int k = 0; k <= m; k++)
						table[0][k] = k;
					for (int j = 1; j <= n; j++)
					{
						for (int k = 1; k <= m; k++)
						{
							int cost = (data_s[id_list[node_child.lid] * M + j - 1] == data_s[node_child.pid * M + k - 1]) ? 0 : 1;
							table[j][k] = 1 + min(table[j - 1][k], table[j][k - 1]);
							table[j][k] = min(table[j - 1][k - 1] + cost, table[j][k]);
						}
					}
					result = table[n][m];
				}
			}
			node_child.min_dis = result;

			if (node_child.size > MAX_SIZE)
			{
				split_list[id_child] = 1;
				node_child.is_leaf = 0;
			}
			else
			{
				split_list[id_child] = 0;
				node_child.is_leaf = 1;
			}

			empty_list[id_child] = 0;
			node_list[id_child] = node_child;
		}
	}
}

__global__ void initIndexData(int *data_info, TN *node_list, int *split_list, int *empty_list, int *id_list)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int total_num = gridDim.x * blockDim.x;

	for (int idx = id; idx < data_info[1]; idx += total_num)
	{
		id_list[idx] = idx;
	}
	if (id == 0)
	{
		node_list[0].size = data_info[1];
		node_list[0].lid = 0;
		node_list[0].pid = -1;
		node_list[0].min_dis = 0;
		node_list[0].is_leaf = 0;
		split_list[0] = 1;
		empty_list[0] = 0;
	}
}

__global__ void showRes(double *dis_list, int size, TN *node_list)
{
	int id = 500;

	printf("%lf\n", dis_list[size - 1]);
	printf("node_list.size : %d\n", node_list[id].size);
	printf("node_list.pid : %d\n", node_list[id].pid);
	printf("node_list.min_dis : %f\n", node_list[id].min_dis);
	printf("node_list.lid : %d\n", node_list[id].lid);
	printf("node_list.is_leaf : %d\n", node_list[id].is_leaf);
}

void indexConstru(short *data_d, char *data_s, int *size_s, int *data_info, int *&id_list, TN *&node_list, int *&max_node_num,
				  int &tree_h, int *&empty_list)
{
	printf("Index construction...\n");

	CHECK(cudaMallocManaged((void **)&max_node_num, sizeof(int)));
	CHECK(cudaMallocManaged((void **)&split_num, sizeof(int)));
	max_node_num[0] = (pow(TREE_ORDER, MAX_H) - 1) / (TREE_ORDER - 1);
	CHECK(cudaMalloc((void **)&split_list, max_node_num[0] * sizeof(int)));
	CHECK(cudaMalloc((void **)&pid_list, max_node_num[0] * sizeof(int)));
	CHECK(cudaMalloc((void **)&dis_list, data_info[1] * sizeof(double)));
	CHECK(cudaMalloc((void **)&empty_list, max_node_num[0] * sizeof(int)));
	CHECK(cudaMalloc((void **)&id_list, data_info[1] * sizeof(int)));
	CHECK(cudaMalloc((void **)&node_list, max_node_num[0] * sizeof(TN)));
	CHECK(cudaMemset(split_list, 0, max_node_num[0] * sizeof(int)));
	CHECK(cudaMemset(empty_list, 1, max_node_num[0] * sizeof(int)));
	split_num[0] = 1;
	cur_level = 0;
	start_idx = 0;
	initIndexData<<<(data_info[1] - 1) / THREAD_NUM + 1, THREAD_NUM>>>(data_info, node_list, split_list,
																	   empty_list, id_list);
	cudaDeviceSynchronize();
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "initIndexData error: %s\n", cudaGetErrorString(cudaStatus));

	// printf("split_num: %d, max_node_num: %d\n", split_num[0], max_node_num[0]);

	while ((cur_level < MAX_H - 1) && (split_num[0] > 0))
	{
		int block_num = pow(TREE_ORDER, cur_level);
		// printf("start_idx: %d\n", start_idx);

		getPivotDis<<<block_num, THREAD_NUM>>>(data_d, data_s, size_s, node_list, split_list, dis_list, id_list,
											   start_idx, data_info, pid_list);
		cudaDeviceSynchronize();
		cudaError_t cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
			fprintf(stderr, "getPivotDis error: %s\n", cudaGetErrorString(cudaStatus));

		thrust::sort_by_key(thrust::device, dis_list, dis_list + data_info[1], id_list);

		nodeSplit<<<block_num, THREAD_NUM>>>(node_list, split_list, dis_list, empty_list, start_idx,
											 pid_list, data_d, id_list, data_info, data_s, size_s);
		cudaDeviceSynchronize();
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
			fprintf(stderr, "getPivotDis error: %s\n", cudaGetErrorString(cudaStatus));

		start_idx += pow(TREE_ORDER, cur_level);
		cur_level++;
		split_num[0] = thrust::reduce(thrust::device, split_list, split_list + max_node_num[0], 0);
	}

	tree_h = cur_level + 1;
	printf("Tree height: %d\n", tree_h);

	cudaFree(dis_list);
	cudaFree(split_list);
	cudaFree(split_num);
	cudaFree(pid_list);
}