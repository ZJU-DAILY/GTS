// Search v1.
// Created on 24-01-05

#pragma once
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <stdio.h>
#include <thrust/reduce.h>
#include "tree.cuh"
#include "config.cuh"

int *query_node_list;
int *search_num;
int *qnode_count;
int *qnode_count_prefix;
int *init_result_id;
float *init_result_dis;
int *qresult_idx;
int *qnode_idx;
int *query_lnode;
int *query_qid;
int *result_num;
int *qp_id;
float *qp_dis;
double *qp_dis_code;
int pnum_level;
int pnum_level_total;
int max_pnum_level;
float *dis_k;
int disk_update;
double *result_dis_code;
int *result_id_idx;
float *result_dis_total;
int *result_id_total;
int qnum_leaf = 500;

__global__ void findNextRnn(int *query_node_list, int start_idx, TN *node_list, float r, short *data_d,
							int *qid_list, int node_num, int *max_node_num, int *data_info, int *empty_list, char *data_s, int *size_s)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int query_id = bid;

	for (int i = tid; i < node_num; i += blockDim.x)
	{
		int nid = start_idx + i;
		int nid_parent = (nid - 1) / TREE_ORDER;

		if ((query_node_list[query_id * max_node_num[0] + nid_parent] == 1) && (empty_list[nid] == 0))
		{
			TN node = node_list[nid];

			float dis_q = 0;
			if (data_info[2] == 2)
			{ // L2 distance
				for (int j = 0; j < data_info[0]; j++)
				{
					dis_q += pow(data_d[node.pid * data_info[0] + j] - data_d[qid_list[query_id] * data_info[0] + j], 2);
				}
				dis_q = pow(dis_q, 0.5);
			}
			else if (data_info[2] == 1)
			{ // L1 distance
				for (int j = 0; j < data_info[0]; j++)
				{
					dis_q += abs(data_d[node.pid * data_info[0] + j] - data_d[qid_list[query_id] * data_info[0] + j]);
				}
			}
			else if (data_info[2] == 0)
			{ // Max value
				float temp = 0;
				for (int j = 0; j < data_info[0]; j++)
				{
					temp = abs(data_d[node.pid * data_info[0] + j] - data_d[qid_list[query_id] * data_info[0] + j]);
					if (temp > dis_q)
						dis_q = temp;
				}
			}
			else if (data_info[2] == 5)
			{
				float sa1 = 0, sa2 = 0, sa3 = 0;
				for (int j = 0; j < data_info[0]; j++)
				{
					sa1 += data_d[node.pid * data_info[0] + j] * data_d[node.pid * data_info[0] + j];
					sa2 += data_d[qid_list[query_id] * data_info[0] + j] * data_d[qid_list[query_id] * data_info[0] + j];
					sa3 += data_d[node.pid * data_info[0] + j] * data_d[qid_list[query_id] * data_info[0] + j];
				}
				sa1 = pow(sa1, 0.5);
				sa2 = pow(sa2, 0.5);
				if (sa1 * sa2 == 0)
				{
					printf("Error!!!\n");
				}
				dis_q = sa3 / (sa1 * sa2);
				if (dis_q > 1)
				{
					dis_q = 0.99999999999999999;
				}
				dis_q = abs(acos(dis_q) * 180 / 3.1415926);
			}
			else if (data_info[2] == 6)
			{
				int n = size_s[node.pid];
				int m = size_s[qid_list[query_id]];
				int table[M][M];
				if (n == 0)
					dis_q = m;
				if (m == 0)
					dis_q = n;
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
							int cost = (data_s[node.pid * M + j - 1] == data_s[qid_list[query_id] * M + k - 1]) ? 0 : 1;
							table[j][k] = 1 + min(table[j - 1][k], table[j][k - 1]);
							table[j][k] = min(table[j - 1][k - 1] + cost, table[j][k]);
						}
					}
					dis_q = table[n][m];
				}
			}

			float dis_lb = node.min_dis - dis_q;
			dis_lb = max(dis_lb, 0.0);
			if (nid % TREE_ORDER != 0)
			{
				float dis_lb2 = dis_q - node_list[nid + 1].min_dis;
				dis_lb = max(dis_lb, dis_lb2);
			}

			if (dis_lb <= r)
				query_node_list[query_id * max_node_num[0] + nid] = 1;
		}
	}
}

__global__ void findNextKnn(int *query_node_list, int start_idx, TN *node_list, float *dis_k, int node_num,
							int *max_node_num, int *empty_list, float *qp_dis, int pnum_level_total, int *qid_list)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int query_id = bid;

	for (int i = tid; i < node_num; i += blockDim.x)
	{
		int nid = start_idx + i;
		int nid_parent = (nid - 1) / TREE_ORDER;
		int pid = i / TREE_ORDER;

		if ((query_node_list[query_id * max_node_num[0] + nid_parent] == 1) && (empty_list[nid] == 0))
		{
			TN node = node_list[nid];

			float dis_q = 0;
			dis_q = qp_dis[query_id * pnum_level_total + pid];

			float dis_lb = node.min_dis - dis_q;
			dis_lb = max(dis_lb, 0.0);
			if (nid % TREE_ORDER != 0)
			{
				float dis_lb2 = dis_q - node_list[nid + 1].min_dis;
				dis_lb = max(dis_lb, dis_lb2);
			}

			if (dis_lb <= dis_k[query_id])
				query_node_list[query_id * max_node_num[0] + nid] = 1;
		}
	}
}

__global__ void updatePnodeFlag(int *query_node_list, int start_idx, int node_num, int *max_node_num, int *empty_list)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int query_id = bid;

	for (int i = tid; i < node_num; i += blockDim.x)
	{
		int nid = start_idx + i;
		int nid_parent = (nid - 1) / TREE_ORDER;

		if ((query_node_list[query_id * max_node_num[0] + nid_parent] == 1) && (empty_list[nid] == 0) && (nid - 1) % TREE_ORDER == 0)
		{
			query_node_list[query_id * max_node_num[0] + nid_parent] = 0;
		}
	}
}

__global__ void updateCnodeFlag(int *query_node_list, int start_idx, int node_num, int *max_node_num, int *empty_list)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int query_id = bid;

	for (int i = tid; i < node_num; i += blockDim.x)
	{
		int nid = start_idx + i;
		int nid_parent = (nid - 1) / TREE_ORDER;

		if ((query_node_list[query_id * max_node_num[0] + nid_parent] == 1) && (empty_list[nid] == 0))
		{
			query_node_list[query_id * max_node_num[0] + nid] = 1;
		}
	}
}

__global__ void getQpDis(int start_idx, TN *node_list, short *data_d, int *qid_list, int *data_info, int *empty_list,
						 int *qp_id, float *qp_dis, double *qp_dis_code, int pnum_level_total, char *data_s, int *size_s)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int query_id = bid;

	for (int i = tid; i < pnum_level_total; i += blockDim.x)
	{
		int nid = start_idx + i * TREE_ORDER;
		double dis_q = INFI_DIS;

		if (empty_list[nid] == 0)
		{
			TN node = node_list[nid];
			dis_q = 0;

			if (data_info[2] == 2)
			{ // L2 distance
				for (int j = 0; j < data_info[0]; j++)
				{
					dis_q += pow(data_d[node.pid * data_info[0] + j] - data_d[qid_list[query_id] * data_info[0] + j], 2);
				}
				dis_q = pow(dis_q, 0.5);
			}
			else if (data_info[2] == 1)
			{ // L1 distance
				for (int j = 0; j < data_info[0]; j++)
				{
					dis_q += abs(data_d[node.pid * data_info[0] + j] - data_d[qid_list[query_id] * data_info[0] + j]);
				}
			}
			else if (data_info[2] == 0)
			{ // Max value
				float temp = 0;
				for (int j = 0; j < data_info[0]; j++)
				{
					temp = abs(data_d[node.pid * data_info[0] + j] - data_d[qid_list[query_id] * data_info[0] + j]);
					if (temp > dis_q)
						dis_q = temp;
				}
			}
			else if (data_info[2] == 5)
			{
				float sa1 = 0, sa2 = 0, sa3 = 0;
				for (int j = 0; j < data_info[0]; j++)
				{
					sa1 += data_d[node.pid * data_info[0] + j] * data_d[node.pid * data_info[0] + j];
					sa2 += data_d[qid_list[query_id] * data_info[0] + j] * data_d[qid_list[query_id] * data_info[0] + j];
					sa3 += data_d[node.pid * data_info[0] + j] * data_d[qid_list[query_id] * data_info[0] + j];
				}
				sa1 = pow(sa1, 0.5);
				sa2 = pow(sa2, 0.5);
				if (sa1 * sa2 == 0)
				{
					printf("Error!!!\n");
				}
				dis_q = sa3 / (sa1 * sa2);
				if (dis_q > 1)
				{
					dis_q = 0.99999999999999999;
				}
				dis_q = abs(acos(dis_q) * 180 / 3.1415926);
			}
			else if (data_info[2] == 6)
			{
				int n = size_s[node.pid];
				int m = size_s[qid_list[query_id]];
				int table[M][M];
				if (n == 0)
					dis_q = m;
				if (m == 0)
					dis_q = n;
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
							int cost = (data_s[node.pid * M + j - 1] == data_s[qid_list[query_id] * M + k - 1]) ? 0 : 1;
							table[j][k] = 1 + min(table[j - 1][k], table[j][k - 1]);
							table[j][k] = min(table[j - 1][k - 1] + cost, table[j][k]);
						}
					}
					dis_q = table[n][m];
				}
			}
		}

		qp_id[query_id * pnum_level_total + i] = i;
		qp_dis[query_id * pnum_level_total + i] = dis_q;
		qp_dis_code[query_id * pnum_level_total + i] = double(dis_q / INFI_DIS + query_id * DIS_CODE);
	}
}

__global__ void updateDisk(int k, int *qp_id, float *qp_dis, float *dis_k, int qnum, int pnum_level_total)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int total_num = gridDim.x * blockDim.x;

	for (int idx = id; idx < qnum; idx += total_num)
	{
		int query_id = idx;
		int i = qp_id[query_id * pnum_level_total + k - 1];

		if (dis_k[query_id] > qp_dis[query_id * pnum_level_total + i])
		{
			dis_k[query_id] = qp_dis[query_id * pnum_level_total + i];
		}
	}
}

__global__ void leafProcessRnn(int *query_lnode, TN *node_list, int *id_list, int *query_qid, short *data_d,
							   int *qid_list, int *init_result_id, float *init_result_dis, int *data_info, float r, int *qresult_idx,
							   int *search_num, char *data_s, int *size_s)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;

	if (bid < search_num[0])
	{
		__shared__ int query_id[1];
		__shared__ TN node[1];

		if (tid == 0)
		{
			query_id[0] = query_qid[bid];
			int nid = query_lnode[bid];
			node[0] = node_list[nid];
		}
		__syncthreads();

		for (int i = tid; i < node[0].size && node[0].is_leaf == 1; i += blockDim.x)
		{
			int data_id = id_list[i + node[0].lid];
			int qid = qid_list[query_id[0]];

			float result = 0;
			if (data_id == qid)
			{
			}
			else if (data_info[2] == 2)
			{ // L2 distance
				for (int j = 0; j < data_info[0]; j++)
				{
					result += pow(data_d[data_id * data_info[0] + j] - data_d[qid * data_info[0] + j], 2);
				}
				result = pow(result, 0.5);
			}
			else if (data_info[2] == 1)
			{ // L1 distance
				for (int j = 0; j < data_info[0]; j++)
				{
					result += abs(data_d[data_id * data_info[0] + j] - data_d[qid * data_info[0] + j]);
				}
			}
			else if (data_info[2] == 0)
			{ // Max value
				float temp = 0;
				for (int j = 0; j < data_info[0]; j++)
				{
					temp = abs(data_d[data_id * data_info[0] + j] - data_d[qid * data_info[0] + j]);
					if (temp > result)
						result = temp;
				}
			}
			else if (data_info[2] == 5)
			{
				float sa1 = 0, sa2 = 0, sa3 = 0;
				for (int j = 0; j < data_info[0]; j++)
				{
					sa1 += data_d[data_id * data_info[0] + j] * data_d[data_id * data_info[0] + j];
					sa2 += data_d[qid * data_info[0] + j] * data_d[qid * data_info[0] + j];
					sa3 += data_d[data_id * data_info[0] + j] * data_d[qid * data_info[0] + j];
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
				int n = size_s[data_id];
				int m = size_s[qid];
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
							int cost = (data_s[data_id * M + j - 1] == data_s[qid * M + k - 1]) ? 0 : 1;
							table[j][k] = 1 + min(table[j - 1][k], table[j][k - 1]);
							table[j][k] = min(table[j - 1][k - 1] + cost, table[j][k]);
						}
					}
					result = table[n][m];
				}
			}

			if (result <= r)
			{
				qresult_idx[bid * MAX_SIZE + i] = 1;
				init_result_id[bid * MAX_SIZE + i] = data_id;
				init_result_dis[bid * MAX_SIZE + i] = result;
				// printf("result: %f\n", result);
			}
		}
	}
}

__global__ void leafProcessKnn(int *query_lnode, TN *node_list, int *id_list, int *query_qid, short *data_d,
							   int *qid_list, int *init_result_id, float *init_result_dis, int *data_info, float *dis_k, int *qresult_idx,
							   int *search_num, char *data_s, int *size_s)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;

	if (bid < search_num[0])
	{
		__shared__ int query_id[1];
		__shared__ TN node[1];

		if (tid == 0)
		{
			query_id[0] = query_qid[bid];
			int nid = query_lnode[bid];
			node[0] = node_list[nid];
		}
		__syncthreads();

		for (int i = tid; i < node[0].size && node[0].is_leaf == 1; i += blockDim.x)
		{
			int data_id = id_list[i + node[0].lid];
			int qid = qid_list[query_id[0]];

			float result = 0;
			if (data_id == qid)
			{
			}
			else if (data_info[2] == 2)
			{ // L2 distance
				for (int j = 0; j < data_info[0]; j++)
				{
					result += pow(data_d[data_id * data_info[0] + j] - data_d[qid * data_info[0] + j], 2);
				}
				result = pow(result, 0.5);
			}
			else if (data_info[2] == 1)
			{ // L1 distance
				for (int j = 0; j < data_info[0]; j++)
				{
					result += abs(data_d[data_id * data_info[0] + j] - data_d[qid * data_info[0] + j]);
				}
			}
			else if (data_info[2] == 0)
			{ // Max value
				float temp = 0;
				for (int j = 0; j < data_info[0]; j++)
				{
					temp = abs(data_d[data_id * data_info[0] + j] - data_d[qid * data_info[0] + j]);
					if (temp > result)
						result = temp;
				}
			}
			else if (data_info[2] == 5)
			{
				float sa1 = 0, sa2 = 0, sa3 = 0;
				for (int j = 0; j < data_info[0]; j++)
				{
					sa1 += data_d[data_id * data_info[0] + j] * data_d[data_id * data_info[0] + j];
					sa2 += data_d[qid * data_info[0] + j] * data_d[qid * data_info[0] + j];
					sa3 += data_d[data_id * data_info[0] + j] * data_d[qid * data_info[0] + j];
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
				int n = size_s[data_id];
				int m = size_s[qid];
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
							int cost = (data_s[data_id * M + j - 1] == data_s[qid * M + k - 1]) ? 0 : 1;
							table[j][k] = 1 + min(table[j - 1][k], table[j][k - 1]);
							table[j][k] = min(table[j - 1][k - 1] + cost, table[j][k]);
						}
					}
					result = table[n][m];
				}
			}

			if (result <= dis_k[query_id[0]])
			{
				qresult_idx[bid * MAX_SIZE + i] = 1;
				init_result_id[bid * MAX_SIZE + i] = data_id;
				init_result_dis[bid * MAX_SIZE + i] = result;
			}
		}
	}
}

__global__ void getQnodeCount(int qnum, int *query_node_list, int *max_node_num, int *qnode_count, int start_q)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int total_num = gridDim.x * blockDim.x;

	for (int idx = id; idx < qnum; idx += total_num)
	{
		int query_id = idx + start_q;
		int start_idx = query_id * max_node_num[0];
		int end_idx = start_idx + max_node_num[0];

		qnode_count[idx] = thrust::reduce(thrust::device, query_node_list + start_idx, query_node_list + end_idx, 0);
	}
}

__global__ void getQresultCount(int qnum, int *qnode_count, int *qnode_count_prefix, int *qresult_count, int *qresult_idx)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int total_num = gridDim.x * blockDim.x;

	for (int idx = id; idx < qnum; idx += total_num)
	{
		int query_id = idx;
		int start_idx = qnode_count_prefix[query_id] * MAX_SIZE;
		int end_idx = (qnode_count_prefix[query_id] + qnode_count[query_id]) * MAX_SIZE;

		qresult_count[query_id] = thrust::reduce(thrust::device, qresult_idx + start_idx, qresult_idx + end_idx, 0);
	}
}

__global__ void mergeLeafNode(int *query_node_list, int *qnode_idx, int *query_lnode, int *max_node_num,
							  int qnum, int *query_qid, int start_q)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int total_num = gridDim.x * blockDim.x;

	for (int idx = id; idx < qnum * max_node_num[0]; idx += total_num)
	{
		if (query_node_list[idx + start_q * max_node_num[0]] == 1)
		{
			int nid = (idx + start_q * max_node_num[0]) % max_node_num[0];
			int qid = (idx + start_q * max_node_num[0]) / max_node_num[0];
			query_lnode[qnode_idx[idx]] = nid;
			query_qid[qnode_idx[idx]] = qid;
		}
	}
}

__global__ void mergeResultRnn(int *qresult_idx, int *init_result_id, float *init_result_dis,
							   int *result_id, float *result_dis, int *search_num)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int total_num = gridDim.x * blockDim.x;

	for (int idx = id; idx < search_num[0] * MAX_SIZE; idx += total_num)
	{
		if (qresult_idx[idx] > 0 && idx == 0)
		{
			int i = qresult_idx[idx];
			result_id[i - 1] = init_result_id[idx];
			result_dis[i - 1] = init_result_dis[idx];
		}
		else if (idx > 0)
		{
			if (qresult_idx[idx] > qresult_idx[idx - 1])
			{
				int i = qresult_idx[idx];
				result_id[i - 1] = init_result_id[idx];
				result_dis[i - 1] = init_result_dis[idx];
			}
		}
	}
}

__global__ void mergeResultKnn(int *qresult_idx, int *init_result_id, float *init_result_dis,
							   int *search_num, int *result_id_total, float *result_dis_total, double *result_dis_code, int *query_qid,
							   int *result_id_idx)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int total_num = gridDim.x * blockDim.x;

	for (int idx = id; idx < search_num[0] * MAX_SIZE; idx += total_num)
	{
		if (qresult_idx[idx] > 0 && idx == 0)
		{
			int i = qresult_idx[idx];
			int j = idx / MAX_SIZE;
			int query_id = query_qid[j];

			result_id_total[i - 1] = init_result_id[idx];
			result_dis_total[i - 1] = init_result_dis[idx];
			result_dis_code[i - 1] = double(double(init_result_dis[idx]) / INFI_DIS + query_id * DIS_CODE);
			result_id_idx[i - 1] = i - 1;
		}
		else if (idx > 0)
		{
			if (qresult_idx[idx] > qresult_idx[idx - 1])
			{
				int i = qresult_idx[idx];
				int j = idx / MAX_SIZE;
				int query_id = query_qid[j];

				result_id_total[i - 1] = init_result_id[idx];
				result_dis_total[i - 1] = init_result_dis[idx];
				result_dis_code[i - 1] = double(double(init_result_dis[idx]) / INFI_DIS + query_id * DIS_CODE);
				result_id_idx[i - 1] = i - 1;
			}
		}
	}
}

__global__ void getKnnResult(int *result_id_total, float *result_dis_total, int *result_id_idx,
							 int *result_id, float *result_dis, int *qresult_count, int *qresult_count_prefix, int k)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int query_id = bid;

	for (int i = tid; i < k; i += blockDim.x)
	{
		int id = qresult_count_prefix[query_id] + i;
		result_id[query_id * k + i] = result_id_total[result_id_idx[id]];
		result_dis[query_id * k + i] = result_dis_total[result_id_idx[id]];
	}
}

__global__ void initQnode(int *query_node_list, int qnum, int *max_node_num)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int total_num = gridDim.x * blockDim.x;
	float num = qnum * max_node_num[0];

	for (int idx = id; idx < num; idx += total_num)
	{
		if (idx % max_node_num[0] == 0)
		{
			query_node_list[idx] = 1;
		}
		else
		{
			query_node_list[idx] = 0;
		}
	}
}

__global__ void initDisk(float *dis_k, int qnum)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int total_num = gridDim.x * blockDim.x;

	for (int idx = id; idx < qnum; idx += total_num)
	{
		dis_k[idx] = INFI_DIS;
	}
}

__global__ void initRes(int *qresult_idx, int num)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int total_num = gridDim.x * blockDim.x;

	for (int idx = id; idx < num; idx += total_num)
	{
		qresult_idx[idx] = 0;
	}
}

__global__ void checkRes(int *qresult_idx, int num)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int total_num = gridDim.x * blockDim.x;

	for (int idx = id; idx < num; idx += total_num)
	{
		if (qresult_idx[idx] == 0)
		{
		}
		else if (qresult_idx[idx] == 1)
		{
			// printf("ok, %d\n", idx);
		}
		else
		{
			printf("qresult_idx[idx]: %d, ", qresult_idx[idx]);
			printf("idx: %d\n", idx);
		}
	}
}

// Range query
void searchIndexRnn(short *data_d, TN *node_list, int *id_list, int *max_node_num, int *qid_list,
					int qnum, float r, int tree_h, int *data_info, int *empty_list, int *&qresult_count,
					int *&qresult_count_prefix, int *&result_id, float *&result_dis, char *data_s, int *size_s,
					float &time_search, FILE *&fcost)
{
	printf("Search...\n");

	auto s = std::chrono::high_resolution_clock::now();
	CHECK(cudaMallocManaged((void **)&search_num, sizeof(int)));
	CHECK(cudaMallocManaged((void **)&result_num, sizeof(int)));
	CHECK(cudaMallocManaged((void **)&qresult_count, qnum_leaf * sizeof(int)));
	CHECK(cudaMallocManaged((void **)&qresult_count_prefix, qnum_leaf * sizeof(int)));
	CHECK(cudaMalloc((void **)&query_node_list, max_node_num[0] * qnum * sizeof(int)));
	CHECK(cudaMalloc((void **)&qnode_idx, max_node_num[0] * qnum_leaf * sizeof(int)));
	CHECK(cudaMalloc((void **)&qnode_count, qnum_leaf * sizeof(int)));
	CHECK(cudaMalloc((void **)&qnode_count_prefix, qnum_leaf * sizeof(int)));
	cur_level = 1;
	start_idx = 1;
	search_num[0] = qnum;
	initQnode<<<(max_node_num[0] * qnum - 1) / THREAD_NUM + 1, THREAD_NUM>>>(query_node_list, qnum, max_node_num);
	cudaDeviceSynchronize();
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "initQnode error: %s\n", cudaGetErrorString(cudaStatus));

	while ((cur_level < tree_h))
	{
		int node_num = pow(TREE_ORDER, cur_level);

		findNextRnn<<<qnum, THREAD_NUM>>>(query_node_list, start_idx, node_list, r, data_d, qid_list, node_num, max_node_num,
										  data_info, empty_list, data_s, size_s);
		cudaDeviceSynchronize();
		cudaError_t cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
			fprintf(stderr, "findNextRnn error: %s\n", cudaGetErrorString(cudaStatus));

		updatePnodeFlag<<<qnum, THREAD_NUM>>>(query_node_list, start_idx, node_num, max_node_num, empty_list);
		cudaDeviceSynchronize();
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
			fprintf(stderr, "updatePnodeFlag error: %s\n", cudaGetErrorString(cudaStatus));

		start_idx += pow(TREE_ORDER, cur_level);
		cur_level++;
	}
	auto e = std::chrono::high_resolution_clock::now();
	std::chrono::duration<float> diff = e - s;
	time_search += diff.count();

	for (int i = 0; i < qnum; i = i + qnum_leaf)
	{
		auto s = std::chrono::high_resolution_clock::now();
		int start_q = i;
		qnum_leaf = min(qnum_leaf, qnum - start_q);

		search_num[0] = thrust::reduce(thrust::device, query_node_list + start_q * max_node_num[0],
									   query_node_list + start_q * max_node_num[0] + max_node_num[0] * qnum_leaf, 0);
		// printf("search num: %d\n", search_num[0]);

		CHECK(cudaMalloc((void **)&init_result_id, search_num[0] * MAX_SIZE * sizeof(int)));
		CHECK(cudaMalloc((void **)&init_result_dis, search_num[0] * MAX_SIZE * sizeof(float)));
		CHECK(cudaMalloc((void **)&qresult_idx, search_num[0] * MAX_SIZE * sizeof(int)));
		CHECK(cudaMalloc((void **)&query_lnode, search_num[0] * sizeof(int)));
		CHECK(cudaMalloc((void **)&query_qid, search_num[0] * sizeof(int)));
		initRes<<<(search_num[0] * MAX_SIZE - 1) / THREAD_NUM + 1, THREAD_NUM>>>(qresult_idx, search_num[0] * MAX_SIZE);
		cudaDeviceSynchronize();
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
			fprintf(stderr, "initRes error: %s\n", cudaGetErrorString(cudaStatus));
		getQnodeCount<<<(qnum_leaf - 1) / THREAD_NUM + 1, THREAD_NUM>>>(qnum_leaf, query_node_list, max_node_num, qnode_count,
																		start_q);
		cudaDeviceSynchronize();
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
			fprintf(stderr, "getQnodeCount error: %s\n", cudaGetErrorString(cudaStatus));
		thrust::exclusive_scan(thrust::device, qnode_count, qnode_count + qnum_leaf, qnode_count_prefix);
		thrust::exclusive_scan(thrust::device, query_node_list + start_q * max_node_num[0],
							   query_node_list + start_q * max_node_num[0] + max_node_num[0] * qnum_leaf, qnode_idx);
		mergeLeafNode<<<(qnum_leaf * max_node_num[0] - 1) / THREAD_NUM + 1, THREAD_NUM>>>(query_node_list, qnode_idx, query_lnode,
																						  max_node_num, qnum_leaf, query_qid, start_q);
		cudaDeviceSynchronize();
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
			fprintf(stderr, "mergeLeafNode error: %s\n", cudaGetErrorString(cudaStatus));

		leafProcessRnn<<<search_num[0], THREAD_NUM>>>(query_lnode, node_list, id_list, query_qid, data_d, qid_list,
													  init_result_id, init_result_dis, data_info, r, qresult_idx, search_num, data_s, size_s);
		cudaDeviceSynchronize();
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
			fprintf(stderr, "leafProcess error: %s\n", cudaGetErrorString(cudaStatus));

		result_num[0] = thrust::reduce(thrust::device, qresult_idx, qresult_idx + (search_num[0] * MAX_SIZE), 0);
		printf("result num: %d\n", result_num[0]);
		CHECK(cudaMallocManaged((void **)&result_id, result_num[0] * sizeof(int)));
		CHECK(cudaMallocManaged((void **)&result_dis, result_num[0] * sizeof(float)));
		getQresultCount<<<(qnum_leaf - 1) / THREAD_NUM + 1, THREAD_NUM>>>(qnum_leaf, qnode_count, qnode_count_prefix,
																		  qresult_count, qresult_idx);
		cudaDeviceSynchronize();
		thrust::exclusive_scan(thrust::device, qresult_count, qresult_count + qnum_leaf, qresult_count_prefix);
		thrust::inclusive_scan(thrust::device, qresult_idx, qresult_idx + search_num[0] * MAX_SIZE, qresult_idx);
		mergeResultRnn<<<(search_num[0] * MAX_SIZE - 1) / THREAD_NUM + 1, THREAD_NUM>>>(qresult_idx, init_result_id,
																						init_result_dis, result_id, result_dis, search_num);
		cudaDeviceSynchronize();
		auto e = std::chrono::high_resolution_clock::now();
		std::chrono::duration<float> diff = e - s;
		time_search += diff.count();

		for (int j = 0; j < qnum_leaf; j++)
		{
			fprintf(fcost, "%d ", qresult_count[j]);
			fflush(fcost);
		}

		s = std::chrono::high_resolution_clock::now();
		cudaFree(init_result_id);
		cudaFree(init_result_dis);
		cudaFree(qresult_idx);
		cudaFree(query_lnode);
		cudaFree(query_qid);
		cudaFree(result_id);
		cudaFree(result_dis);
		e = std::chrono::high_resolution_clock::now();
		diff = e - s;
		time_search += diff.count();
	}

	s = std::chrono::high_resolution_clock::now();
	cudaFree(query_node_list);
	cudaFree(search_num);
	cudaFree(result_num);
	cudaFree(qnode_count);
	cudaFree(qnode_count_prefix);
	cudaFree(qnode_idx);
	cudaFree(qresult_count);
	cudaFree(qresult_count_prefix);
	e = std::chrono::high_resolution_clock::now();
	diff = e - s;
	time_search += diff.count();
}

// knn query
void searchIndexKnn(short *data_d, TN *node_list, int *id_list, int *max_node_num, int *qid_list,
					int qnum, int k, int tree_h, int *data_info, int *empty_list, int *&result_id, float *&result_dis,
					int *&qresult_count, int *&qresult_count_prefix, char *data_s, int *size_s, float &time_search, FILE *fcost)
{
	printf("Search...\n");

	auto s = std::chrono::high_resolution_clock::now();
	max_pnum_level = pow(TREE_ORDER, tree_h - 2);
	CHECK(cudaMallocManaged((void **)&search_num, sizeof(int)));
	CHECK(cudaMallocManaged((void **)&result_num, sizeof(int)));
	CHECK(cudaMallocManaged((void **)&result_id, k * qnum_leaf * sizeof(int)));
	CHECK(cudaMallocManaged((void **)&result_dis, k * qnum_leaf * sizeof(float)));
	CHECK(cudaMalloc((void **)&qresult_count, qnum_leaf * sizeof(int)));
	CHECK(cudaMalloc((void **)&qresult_count_prefix, qnum_leaf * sizeof(int)));
	CHECK(cudaMalloc((void **)&query_node_list, max_node_num[0] * qnum * sizeof(int)));
	CHECK(cudaMalloc((void **)&qnode_idx, max_node_num[0] * qnum_leaf * sizeof(int)));
	CHECK(cudaMalloc((void **)&qnode_count, qnum_leaf * sizeof(int)));
	CHECK(cudaMalloc((void **)&qnode_count_prefix, qnum_leaf * sizeof(int)));
	CHECK(cudaMalloc((void **)&qp_dis, max_pnum_level * qnum * sizeof(float)));
	CHECK(cudaMalloc((void **)&qp_id, max_pnum_level * qnum * sizeof(int)));
	CHECK(cudaMalloc((void **)&qp_dis_code, max_pnum_level * qnum * sizeof(double)));
	CHECK(cudaMalloc((void **)&dis_k, qnum * sizeof(float)));
	cur_level = 1;
	start_idx = 1;
	search_num[0] = qnum;
	disk_update = 0;
	initQnode<<<(max_node_num[0] * qnum - 1) / THREAD_NUM + 1, THREAD_NUM>>>(query_node_list, qnum, max_node_num);
	cudaDeviceSynchronize();
	initDisk<<<(qnum - 1) / THREAD_NUM + 1, THREAD_NUM>>>(dis_k, qnum);
	cudaDeviceSynchronize();

	while ((cur_level < tree_h))
	{
		int node_num = pow(TREE_ORDER, cur_level);
		pnum_level = node_num - thrust::reduce(thrust::device, empty_list + start_idx, empty_list + start_idx + node_num, 0);
		pnum_level = pnum_level / TREE_ORDER;
		printf("pnum level: %d\n", pnum_level);
		pnum_level_total = node_num / TREE_ORDER;
		printf("pnum level total: %d\n", pnum_level_total);

		if (disk_update == 0 && pnum_level < k)
		{
			updateCnodeFlag<<<qnum, THREAD_NUM>>>(query_node_list, start_idx, node_num, max_node_num, empty_list);
			cudaDeviceSynchronize();
		}
		else
		{
			disk_update = 1;

			if (pnum_level >= k)
			{
				getQpDis<<<qnum, THREAD_NUM>>>(start_idx, node_list, data_d, qid_list, data_info, empty_list,
											   qp_id, qp_dis, qp_dis_code, pnum_level_total, data_s, size_s);
				cudaDeviceSynchronize();

				thrust::sort_by_key(thrust::device, qp_dis_code, qp_dis_code + pnum_level_total * qnum, qp_id);

				updateDisk<<<(qnum - 1) / THREAD_NUM + 1, THREAD_NUM>>>(k, qp_id, qp_dis, dis_k, qnum, pnum_level_total);
				cudaDeviceSynchronize();
			}

			findNextKnn<<<qnum, THREAD_NUM>>>(query_node_list, start_idx, node_list, dis_k, node_num, max_node_num,
											  empty_list, qp_dis, pnum_level_total, qid_list);
			cudaDeviceSynchronize();
			cudaError_t cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess)
				fprintf(stderr, "findNextKnn error: %s\n", cudaGetErrorString(cudaStatus));
		}
		updatePnodeFlag<<<qnum, THREAD_NUM>>>(query_node_list, start_idx, node_num, max_node_num, empty_list);
		cudaDeviceSynchronize();

		start_idx += pow(TREE_ORDER, cur_level);
		cur_level++;
	}
	auto e = std::chrono::high_resolution_clock::now();
	std::chrono::duration<float> diff = e - s;
	time_search += diff.count();

	for (int i = 0; i < qnum; i = i + qnum_leaf)
	{
		auto s = std::chrono::high_resolution_clock::now();
		int start_q = i;
		qnum_leaf = min(qnum_leaf, qnum - start_q);

		search_num[0] = thrust::reduce(thrust::device, query_node_list + start_q * max_node_num[0],
									   query_node_list + start_q * max_node_num[0] + max_node_num[0] * qnum_leaf, 0);
		// printf("search num: %d\n", search_num[0]);

		CHECK(cudaMalloc((void **)&init_result_id, search_num[0] * MAX_SIZE * sizeof(int)));
		CHECK(cudaMalloc((void **)&init_result_dis, search_num[0] * MAX_SIZE * sizeof(float)));
		CHECK(cudaMalloc((void **)&qresult_idx, search_num[0] * MAX_SIZE * sizeof(int)));
		CHECK(cudaMalloc((void **)&query_lnode, search_num[0] * sizeof(int)));
		CHECK(cudaMalloc((void **)&query_qid, search_num[0] * sizeof(int)));
		initRes<<<(search_num[0] * MAX_SIZE - 1) / THREAD_NUM + 1, THREAD_NUM>>>(qresult_idx, search_num[0] * MAX_SIZE);
		cudaDeviceSynchronize();
		cudaError_t cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
			fprintf(stderr, "initRes error: %s\n", cudaGetErrorString(cudaStatus));
		getQnodeCount<<<(qnum_leaf - 1) / THREAD_NUM + 1, THREAD_NUM>>>(qnum_leaf, query_node_list, max_node_num, qnode_count,
																		start_q);
		cudaDeviceSynchronize();
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
			fprintf(stderr, "getQnodeCount error: %s\n", cudaGetErrorString(cudaStatus));
		thrust::exclusive_scan(thrust::device, qnode_count, qnode_count + qnum_leaf, qnode_count_prefix);
		thrust::exclusive_scan(thrust::device, query_node_list + start_q * max_node_num[0],
							   query_node_list + start_q * max_node_num[0] + max_node_num[0] * qnum_leaf, qnode_idx);
		mergeLeafNode<<<(qnum_leaf * max_node_num[0] - 1) / THREAD_NUM + 1, THREAD_NUM>>>(query_node_list, qnode_idx, query_lnode,
																						  max_node_num, qnum_leaf, query_qid, start_q);
		cudaDeviceSynchronize();
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
			fprintf(stderr, "mergeLeafNode error: %s\n", cudaGetErrorString(cudaStatus));

		leafProcessKnn<<<search_num[0], THREAD_NUM>>>(query_lnode, node_list, id_list, query_qid, data_d, qid_list,
													  init_result_id, init_result_dis, data_info, dis_k, qresult_idx, search_num, data_s, size_s);
		cudaDeviceSynchronize();
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
			fprintf(stderr, "leafProcess error: %s\n", cudaGetErrorString(cudaStatus));

		result_num[0] = thrust::reduce(thrust::device, qresult_idx, qresult_idx + (search_num[0] * MAX_SIZE), 0);
		// printf("result num: %d\n", result_num[0]);
		CHECK(cudaMalloc((void **)&result_id_total, result_num[0] * sizeof(int)));
		CHECK(cudaMalloc((void **)&result_id_idx, result_num[0] * sizeof(int)));
		CHECK(cudaMalloc((void **)&result_dis_total, result_num[0] * sizeof(float)));
		CHECK(cudaMalloc((void **)&result_dis_code, result_num[0] * sizeof(double)));
		getQresultCount<<<(qnum_leaf - 1) / THREAD_NUM + 1, THREAD_NUM>>>(qnum_leaf, qnode_count, qnode_count_prefix,
																		  qresult_count, qresult_idx);
		cudaDeviceSynchronize();
		thrust::exclusive_scan(thrust::device, qresult_count, qresult_count + qnum_leaf, qresult_count_prefix);
		thrust::inclusive_scan(thrust::device, qresult_idx, qresult_idx + search_num[0] * MAX_SIZE, qresult_idx);
		mergeResultKnn<<<(search_num[0] * MAX_SIZE - 1) / THREAD_NUM + 1, THREAD_NUM>>>(qresult_idx,
																						init_result_id, init_result_dis, search_num, result_id_total, result_dis_total, result_dis_code,
																						query_qid, result_id_idx);
		cudaDeviceSynchronize();
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
			fprintf(stderr, "mergeResultKnn error: %s\n", cudaGetErrorString(cudaStatus));

		int rnum = result_num[0];
		thrust::sort_by_key(thrust::device, result_dis_code, result_dis_code + rnum, result_id_idx);
		getKnnResult<<<qnum_leaf, THREAD_NUM>>>(result_id_total, result_dis_total, result_id_idx,
												result_id, result_dis, qresult_count, qresult_count_prefix, k);
		cudaDeviceSynchronize();
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
			fprintf(stderr, "getKnnResult error: %s\n", cudaGetErrorString(cudaStatus));
		auto e = std::chrono::high_resolution_clock::now();
		std::chrono::duration<float> diff = e - s;
		time_search += diff.count();

		for (int j = 0; j < qnum_leaf; j++)
		{
			fprintf(fcost, "%f ", result_dis[j * k + k - 1]);
			fflush(fcost);
		}

		s = std::chrono::high_resolution_clock::now();
		cudaFree(init_result_id);
		cudaFree(init_result_dis);
		cudaFree(qresult_idx);
		cudaFree(query_lnode);
		cudaFree(query_qid);
		cudaFree(result_id_total);
		cudaFree(result_dis_total);
		cudaFree(result_dis_code);
		cudaFree(result_id_idx);
		e = std::chrono::high_resolution_clock::now();
		diff = e - s;
		time_search += diff.count();
	}

	s = std::chrono::high_resolution_clock::now();
	cudaFree(query_node_list);
	cudaFree(search_num);
	cudaFree(result_num);
	cudaFree(qnode_count);
	cudaFree(qnode_count_prefix);
	cudaFree(qnode_idx);
	cudaFree(qp_dis);
	cudaFree(qp_id);
	cudaFree(qp_dis_code);
	cudaFree(dis_k);
	cudaFree(qresult_count);
	cudaFree(qresult_count_prefix);
	cudaFree(result_id);
	cudaFree(result_dis);
	e = std::chrono::high_resolution_clock::now();
	diff = e - s;
	time_search += diff.count();
}