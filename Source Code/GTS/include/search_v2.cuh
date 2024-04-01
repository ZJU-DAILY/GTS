// Search v2. Dynamically adjust memory allocation
// Created on 24-01-05

#pragma once
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <stdio.h>
#include <thrust/reduce.h>
#include <stack>
#include <thrust/count.h>
#include <cuda_runtime.h>
#include "file.cuh"
#include "tree.cuh"
#include "config.cuh"
using namespace std;

stack<int> st;			  // Instead of recursive calls, control the query interval of each layer (qs, qe, cur_level, qnum_up, offset_n, qs_up, size_a).
int *res;				  // The output result of range query.
float *res_dis;			  // The output result of knn query.
int qs, qe;				  // The start id and the end id of query at each process.
int *p_list;			  // The total list of queries to process for all layers.
double *p_list_k;		  // The total list of queries to process for all layers for knn queries.
int size_avg;			  // The average size of remaining levels.
int *size_list;			  // The real size of each level.
int qnum_l;				  // The number of queries that can be processed simultaneously at the current layer.
int qnum_up;			  // The number of queries that can be processed simultaneously at the upper layer.
int nnum_l;				  // The number of nodes at the current level.
int size_a;				  // The available size.
int offset_p;			  // The offset of the starting position of the p_list at current layer.
int offset_up_p;		  // The offset of the starting position of the p_list at upper layer.
int offset_n;			  // The offset of the starting position of the node_list at current layer.
int qs_up;				  // The start id of query at upper layer.
float *disk;			  // The distance to the current k-th neighbor.
bool update_disk = false; // The flag to determine whether the disk has been updated.
float input_size = 0;

// Process the nodes of the current layer and determine if the node will be pruned.
// A thread is assigned for a (query, node) pair.
__global__ void nodeProcessRnn(TN *node_list, float r, short *data_d, int *qid_list, int *data_info, int *empty_list,
							   char *data_s, int *size_s, int qnum_l, int qnum_up, int nnum_l, int *p_list, int offset_p, int offset_up_p, int offset_n,
							   int qs, int qs_up)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int total_num = gridDim.x * blockDim.x;

	for (int i = tid; i < nnum_l * qnum_l; i += total_num)
	{
		int qid_p = i % qnum_l;				   // Query idx in p_list
		int qid_p_up = qid_p + qs - qs_up;	   // Query idx at upper layer in p_list
		int qid = qid_p + qs;				   // Query idx in qid_lsit
		int nid_p = i / qnum_l;				   // Node idx in p_list at current level.
		int nid_parent_p = nid_p / TREE_ORDER; // Parent node idx in p_list at upper level.
		int nid = nid_p + offset_n;			   // Node idx in node_list

		// Reset p_list
		p_list[offset_p + i] = 0;

		if (p_list[offset_up_p + nid_parent_p * qnum_up + qid_p_up] == 1 && (empty_list[nid] == 0))
		{
			TN node = node_list[nid];

			float dis_q = 0;
			if (data_info[2] == 2)
			{ // L2 distance
				for (int j = 0; j < data_info[0]; j++)
				{
					dis_q += pow(data_d[node.pid * data_info[0] + j] - data_d[qid_list[qid] * data_info[0] + j], 2);
				}
				dis_q = pow(dis_q, 0.5);
			}
			else if (data_info[2] == 1)
			{ // L1 distance
				for (int j = 0; j < data_info[0]; j++)
				{
					dis_q += abs(data_d[node.pid * data_info[0] + j] - data_d[qid_list[qid] * data_info[0] + j]);
				}
			}
			else if (data_info[2] == 0)
			{ // Max value
				float temp = 0;
				for (int j = 0; j < data_info[0]; j++)
				{
					temp = abs(data_d[node.pid * data_info[0] + j] - data_d[qid_list[qid] * data_info[0] + j]);
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
					sa2 += data_d[qid_list[qid] * data_info[0] + j] * data_d[qid_list[qid] * data_info[0] + j];
					sa3 += data_d[node.pid * data_info[0] + j] * data_d[qid_list[qid] * data_info[0] + j];
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
				int m = size_s[qid_list[qid]];
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
							int cost = (data_s[node.pid * M + j - 1] == data_s[qid_list[qid] * M + k - 1]) ? 0 : 1;
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
				p_list[offset_p + i] = 1;
		}
	}
}

// Process the nodes of the current layer and determine if the node will be pruned.
// A thread is assigned for a (query, node) pair.
__global__ void nodeProcessKnn(TN *node_list, float *disk, int *empty_list, int qnum_l, int qnum_up, int nnum_l, double *p_list_k,
							   int offset_p, int offset_up_p, int offset_n, int qs, int qs_up)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int total_num = gridDim.x * blockDim.x;

	for (int i = tid; i < nnum_l * qnum_l; i += total_num)
	{
		int qid_p = i % qnum_l;				   // Query idx in p_list
		int qid_p_up = qid_p + qs - qs_up;	   // Query idx at upper layer in p_list
		int qid = qid_p + qs;				   // Query idx in qid_lsit
		int nid_p = (i / qnum_l);			   // The node idx in p_list at current level.
		int nid_parent_p = nid_p / TREE_ORDER; // Parent node idx in p_list at upper level.
		int nid = nid_p + offset_n;			   // Node idx in node_list
		int temp = (nnum_l / TREE_ORDER / TREE_ORDER * 3);
		int ofst_up = temp * qnum_up;				   // Offset at upper level.
		int ofst = (nnum_l / TREE_ORDER * 3) * qnum_l; // Offset at current level.
		int idx_p = nid_parent_p * qnum_l + qid_p;	   // The index of position that holds the distance between pivot and query.

		// Reset p_list
		p_list_k[offset_p + ofst + i] = 0;

		if (p_list_k[offset_up_p + ofst_up + nid_parent_p * qnum_up + qid_p_up] == 1 && (empty_list[nid] == 0))
		{
			TN node = node_list[nid];

			float dis_q = p_list_k[offset_p + ofst / 3 + idx_p];

			float dis_lb = node.min_dis - dis_q;
			dis_lb = max(dis_lb, 0.0);

			if (nid % TREE_ORDER != 0)
			{
				float dis_lb2 = dis_q - node_list[nid + 1].min_dis;
				dis_lb = max(dis_lb, dis_lb2);
			}

			if (dis_lb <= disk[qid])
				p_list_k[offset_p + ofst + i] = 1;
		}
	}
}

// Initialize p_list.
__global__ void initPList(int *p_list, int qnum)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int total_num = gridDim.x * blockDim.x;

	for (int i = tid; i < qnum; i += total_num)
	{
		p_list[i] = 1;
	}
}

// Initialize p_list.
__global__ void initPListKnn(double *p_list_k, int qnum)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int total_num = gridDim.x * blockDim.x;

	for (int i = tid; i < qnum; i += total_num)
	{
		p_list_k[i] = 1;
	}
}

// Get counts of query.
__global__ void getQCount(int ls, int le, int *p_list, int offset_up_p, int offset_p, int qnum_up, int nnum_up)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int total_num = gridDim.x * blockDim.x;

	for (int i = tid; i < qnum_up * nnum_up; i += total_num)
	{
		int qid_p = i % qnum_up; // Query idx in p_list
		int nid_p = i / qnum_up; // Node idx in p_list at current level.

		if (p_list[offset_up_p + i] == 1 && qid_p >= ls && qid_p < le)
		{
			p_list[offset_p + (qid_p - ls) * nnum_up + nid_p] = 1;
		}
		else if (p_list[offset_up_p + i] == 0 && qid_p >= ls && qid_p < le)
		{
			p_list[offset_p + (qid_p - ls) * nnum_up + nid_p] = 0;
		}
	}
}

// Get counts of query for knn.
__global__ void getQCountKnn(int ls, int le, double *p_list_k, int offset_up_p, int offset_p, int qnum_up, int nnum_up)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int total_num = gridDim.x * blockDim.x;
	int ofst_up = (nnum_up / TREE_ORDER * 3) * qnum_up; // Offset at upper level.

	for (int i = tid; i < qnum_up * nnum_up; i += total_num)
	{
		int qid_p = i % qnum_up; // Query idx in p_list
		int nid_p = i / qnum_up; // Node idx in p_list at current level.

		if (p_list_k[offset_up_p + ofst_up + i] == 1 && qid_p >= ls && qid_p < le)
		{
			p_list_k[offset_p + (qid_p - ls) * nnum_up + nid_p] = 1;
		}
		else if (p_list_k[offset_up_p + ofst_up + i] == 0 && qid_p >= ls && qid_p < le)
		{
			p_list_k[offset_p + (qid_p - ls) * nnum_up + nid_p] = 0;
		}
	}
}

// Merge leaf node.
// A thread is assigned for a (query, node) pair.
__global__ void mergeLNode(int ls, int le, int *p_list, int offset_up_p, int offset_up_n, int qs_up, int offset_p, int *size_list,
						   int cur_level, int qnum_up, int nnum_up)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int total_num = gridDim.x * blockDim.x;

	for (int i = tid; i < qnum_up * nnum_up; i += total_num)
	{
		int qid_p = i % qnum_up;	   // Query idx in p_list
		int qid = qid_p + qs_up;	   // Query idx in qid_lsit
		int nid_p = i / qnum_up;	   // Node idx in p_list at current level.
		int nid = nid_p + offset_up_n; // Node idx in node_list

		// Merge leaf node.
		if (p_list[offset_up_p + i] == 1 && qid_p >= ls && qid_p < le)
		{
			int idx_pre = p_list[offset_p + (qid_p - ls) * nnum_up + nid_p];			  // Idx in prefix sum list
			p_list[offset_p + size_list[cur_level] / (MAX_SIZE + 3) + idx_pre] = nid;	  // Merge the real node ID in plist
			p_list[offset_p + size_list[cur_level] / (MAX_SIZE + 3) * 2 + idx_pre] = qid; // Merge the real query ID in plist
		}
	}
}

// Merge leaf nodes for knn.
// A thread is assigned for a (query, node) pair.
__global__ void mergeLNodeKnn(int ls, int le, double *p_list_k, int offset_up_p, int offset_up_n, int qs_up, int offset_p, int *size_list,
							  int cur_level, int qnum_up, int nnum_up)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int total_num = gridDim.x * blockDim.x;
	int ofst_up = (nnum_up / TREE_ORDER * 3) * qnum_up; // Offset at upper level.

	for (int i = tid; i < qnum_up * nnum_up; i += total_num)
	{
		int qid_p = i % qnum_up;	   // Query idx in p_list
		int qid = qid_p + qs_up;	   // Query idx in qid_lsit
		int nid_p = i / qnum_up;	   // Node idx in p_list at current level.
		int nid = nid_p + offset_up_n; // Node idx in node_list

		// Merge leaf node.
		if (p_list_k[offset_up_p + ofst_up + i] == 1 && qid_p >= ls && qid_p < le)
		{
			int idx_pre = p_list_k[offset_p + (qid_p - ls) * nnum_up + nid_p];					// Idx in prefix sum list
			p_list_k[offset_p + size_list[cur_level] / (MAX_SIZE * 3 + 3) + idx_pre] = nid;		// Merge the real node ID in plist
			p_list_k[offset_p + size_list[cur_level] / (MAX_SIZE * 3 + 3) * 2 + idx_pre] = qid; // Merge the real query ID in plist
		}
	}
}

// Process the nodes of the current layer and determine if the node will be pruned.
__global__ void dataProcessRnn(TN *node_list, float r, short *data_d, int *qid_list, int *data_info, char *data_s, int *size_s,
							   int *p_list, int offset_p, int *id_list, int cur_level, int *size_list, int nnum_up)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;

	int nid_p = bid;																// Node idx in p_list.
	int qid_p = bid;																// Query idx in p_list.
	int did = tid;																	// Data idx in the leaf node.
	int nid = p_list[offset_p + size_list[cur_level] / (MAX_SIZE + 3) + nid_p];		// Node idx in node_list
	int qid = p_list[offset_p + size_list[cur_level] / (MAX_SIZE + 3) * 2 + qid_p]; // Query idx in qid_lsit
	TN node = node_list[nid];														// Leaf node

	if (did < node.size && node.is_leaf == 1)
	{
		int data_id = id_list[node.lid + did]; // Data idx in dataset.
		// printf("data_id: %d\n", data_id);

		float result = 0;
		if (data_id == qid_list[qid])
		{
		}
		else if (data_info[2] == 2)
		{ // L2 distance
			for (int j = 0; j < data_info[0]; j++)
			{
				result += pow(data_d[data_id * data_info[0] + j] - data_d[qid_list[qid] * data_info[0] + j], 2);
			}
			result = pow(result, 0.5);
		}
		else if (data_info[2] == 1)
		{ // L1 distance
			for (int j = 0; j < data_info[0]; j++)
			{
				result += abs(data_d[data_id * data_info[0] + j] - data_d[qid_list[qid] * data_info[0] + j]);
			}
		}
		else if (data_info[2] == 0)
		{ // Max value
			float temp = 0;
			for (int j = 0; j < data_info[0]; j++)
			{
				temp = abs(data_d[data_id * data_info[0] + j] - data_d[qid_list[qid] * data_info[0] + j]);
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
				sa2 += data_d[qid_list[qid] * data_info[0] + j] * data_d[qid_list[qid] * data_info[0] + j];
				sa3 += data_d[data_id * data_info[0] + j] * data_d[qid_list[qid] * data_info[0] + j];
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
			int m = size_s[qid_list[qid]];
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
						int cost = (data_s[data_id * M + j - 1] == data_s[qid_list[qid] * M + k - 1]) ? 0 : 1;
						table[j][k] = 1 + min(table[j - 1][k], table[j][k - 1]);
						table[j][k] = min(table[j - 1][k - 1] + cost, table[j][k]);
					}
				}
				result = table[n][m];
			}
		}

		if (result <= r)
		{
			// p_list[2 * offset_p - offset_up_p + 2 * lnum + i] = qid + 1;
			// atomicAdd(&res[qid], 1);
			p_list[offset_p + size_list[cur_level] / (MAX_SIZE + 3) * 3 + bid * MAX_SIZE + did] = 1;
		}
		else
			p_list[offset_p + size_list[cur_level] / (MAX_SIZE + 3) * 3 + bid * MAX_SIZE + did] = 0;
	}

	else if (did < MAX_SIZE)
	{
		p_list[offset_p + size_list[cur_level] / (MAX_SIZE + 3) * 3 + bid * MAX_SIZE + did] = 0;
	}
}

// Process the nodes of the current layer and determine if the node will be pruned.
__global__ void dataProcessKnn(TN *node_list, float *disk, short *data_d, int *qid_list, int *data_info, char *data_s, int *size_s,
							   double *p_list_k, int offset_p, int *id_list, int cur_level, int *size_list, int nnum_up)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;

	int nid_p = bid;																	  // Node idx in p_list.
	int qid_p = bid;																	  // Query idx in p_list.
	int nid = p_list_k[offset_p + size_list[cur_level] / (MAX_SIZE * 3 + 3) + nid_p];	  // Node idx in node_list
	int qid = p_list_k[offset_p + size_list[cur_level] / (MAX_SIZE * 3 + 3) * 2 + qid_p]; // Query idx in qid_lsit
	TN node = node_list[nid];															  // Leaf node

	for (int did = tid; did < MAX_SIZE; did += THREAD_NUM)
	{
		double result = INFI_DIS;

		if (did < node.size && node.is_leaf == 1)
		{
			int data_id = id_list[node.lid + did]; // Data idx in dataset.

			result = 0;
			if (data_id == qid_list[qid])
			{
			}
			else if (data_info[2] == 2)
			{ // L2 distance
				for (int j = 0; j < data_info[0]; j++)
				{
					result += pow(data_d[data_id * data_info[0] + j] - data_d[qid_list[qid] * data_info[0] + j], 2);
				}
				result = pow(result, 0.5);
			}
			else if (data_info[2] == 1)
			{ // L1 distance
				for (int j = 0; j < data_info[0]; j++)
				{
					result += abs(data_d[data_id * data_info[0] + j] - data_d[qid_list[qid] * data_info[0] + j]);
				}
			}
			else if (data_info[2] == 0)
			{ // Max value
				float temp = 0;
				for (int j = 0; j < data_info[0]; j++)
				{
					temp = abs(data_d[data_id * data_info[0] + j] - data_d[qid_list[qid] * data_info[0] + j]);
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
					sa2 += data_d[qid_list[qid] * data_info[0] + j] * data_d[qid_list[qid] * data_info[0] + j];
					sa3 += data_d[data_id * data_info[0] + j] * data_d[qid_list[qid] * data_info[0] + j];
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
				int m = size_s[qid_list[qid]];
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
							int cost = (data_s[data_id * M + j - 1] == data_s[qid_list[qid] * M + k - 1]) ? 0 : 1;
							table[j][k] = 1 + min(table[j - 1][k], table[j][k - 1]);
							table[j][k] = min(table[j - 1][k - 1] + cost, table[j][k]);
						}
					}
					result = table[n][m];
				}
			}

			if (result > disk[qid])
			{
				result = INFI_DIS;
			}
		}

		// Save result.
		p_list_k[offset_p + size_list[cur_level] / (MAX_SIZE * 3 + 3) * 3 + bid * MAX_SIZE + did] = bid * MAX_SIZE + did;
		p_list_k[offset_p + size_list[cur_level] / (MAX_SIZE * 3 + 3) * (3 + MAX_SIZE) + bid * MAX_SIZE + did] = result;
		p_list_k[offset_p + size_list[cur_level] / (MAX_SIZE * 3 + 3) * (3 + 2 * MAX_SIZE) + bid * MAX_SIZE + did] =
			double(result / INFI_DIS + qid * DIS_CODE);
	}
}

// Merge result.
__global__ void mergeResRnn(int ls, int le, int *p_list, int offset_p, int *size_list, int cur_level, int nnum_up, int lnum, int *res)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int total_num = gridDim.x * blockDim.x;

	for (int i = tid; i < (le - ls); i += total_num)
	{
		int s = offset_p + size_list[cur_level] / (MAX_SIZE + 3) * 3 + p_list[offset_p + i * nnum_up] * MAX_SIZE;
		int e;
		if (i < le - ls - 1)
		{
			e = offset_p + size_list[cur_level] / (MAX_SIZE + 3) * 3 + p_list[offset_p + (i + 1) * nnum_up] * MAX_SIZE;
		}
		else
		{
			e = offset_p + size_list[cur_level] / (MAX_SIZE + 3) * 3 + lnum * MAX_SIZE;
		}

		int qid = p_list[offset_p + size_list[cur_level] / (MAX_SIZE + 3) * 2 + p_list[offset_p + i * nnum_up]];
		int num = thrust::reduce(thrust::device, p_list + s, p_list + e, 0);
		res[qid] = num;
	}
}

// Merge result.
__global__ void mergeResKnn(int ls, int le, double *p_list_k, int offset_p, int *size_list, int cur_level, int nnum_up, float *res_dis, int k)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int total_num = gridDim.x * blockDim.x;

	for (int i = tid; i < (le - ls); i += total_num)
	{
		int s = offset_p + size_list[cur_level] / (MAX_SIZE * 3 + 3) * 3 + p_list_k[offset_p + i * nnum_up] * MAX_SIZE;
		int idx = p_list_k[s + k - 1];
		int idx_q = offset_p + size_list[cur_level] / (MAX_SIZE * 3 + 3) * 2 + p_list_k[offset_p + i * nnum_up];
		int qid = p_list_k[idx_q];
		res_dis[qid] = p_list_k[offset_p + size_list[cur_level] / (MAX_SIZE * 3 + 3) * (3 + MAX_SIZE) + idx];
	}
}

// Initialize res.
__global__ void initResV2(int *res, int qnum)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int total_num = gridDim.x * blockDim.x;

	for (int i = tid; i < qnum; i += total_num)
	{
		res[i] = 0;
	}
}

// Initialize disk.
__global__ void initDisK(float *disk, int qnum)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int total_num = gridDim.x * blockDim.x;

	for (int i = tid; i < qnum; i += total_num)
	{
		disk[i] = INFI_DIS;
	}
}

// Label child nodes without varification.
__global__ void labelCNode(int *empty_list, int qnum_l, int qnum_up, int nnum_l, double *p_list_k, int offset_p, int offset_up_p, int offset_n,
						   int qs, int qs_up)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int total_num = gridDim.x * blockDim.x;
	int ofst = (nnum_l / TREE_ORDER * 3) * qnum_l; // Offset at current level.
	int temp = (nnum_l / TREE_ORDER / TREE_ORDER * 3);
	int ofst_up = temp * qnum_up; // Offset at upper level.

	/*if (tid == 0) {
		printf("ofst: %d\n", ofst);
		printf("ofst_up: %d\n", ofst_up);
	}*/

	for (int i = tid; i < nnum_l * qnum_l; i += total_num)
	{
		int qid_p = i % qnum_l;				   // Query idx in p_list
		int qid_p_up = qid_p + qs - qs_up;	   // Query idx at upper layer in p_list
		int nid_p = i / qnum_l;				   // Node idx in p_list at current level.
		int nid_parent_p = nid_p / TREE_ORDER; // Parent node idx in p_list at upper level.
		int nid = nid_p + offset_n;			   // Node idx in node_list

		// Reset p_list
		p_list_k[offset_p + ofst + i] = 0;

		if (p_list_k[offset_up_p + ofst_up + nid_parent_p * qnum_up + qid_p_up] == 1 && (empty_list[nid] == 0))
		{
			p_list_k[offset_p + ofst + i] = 1;
		}
	}
}

// Compute the distances between pivots and queries at current level.
__global__ void getDisPQ(TN *node_list, short *data_d, int *qid_list, int *data_info, int *empty_list, char *data_s,
						 int *size_s, int qnum_l, int qnum_up, int nnum_l, double *p_list_k, int offset_p, int offset_up_p, int offset_n,
						 int qs, int qs_up, int pnum_level_total)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int total_num = gridDim.x * blockDim.x;

	for (int i = tid; i < pnum_level_total * qnum_l; i += total_num)
	{
		int qid_p = i % qnum_l;						   // Query idx in p_list
		int qid_p_up = qid_p + qs - qs_up;			   // Query idx at upper layer in p_list
		int qid = qid_p + qs;						   // Query idx in qid_lsit
		int nid_p = (i / qnum_l) * TREE_ORDER;		   // The first node idx in p_list at current level.
		int nid_parent_p = nid_p / TREE_ORDER;		   // Parent node idx in p_list at upper level.
		int nid = nid_p + offset_n;					   // Node idx in node_list
		int ofst = (nnum_l / TREE_ORDER * 3) * qnum_l; // Offset at current level.
		int temp = (nnum_l / TREE_ORDER / TREE_ORDER * 3);
		int ofst_up = temp * qnum_up; // Offset at upper level.

		double dis_q = INFI_DIS;

		if (p_list_k[offset_up_p + ofst_up + nid_parent_p * qnum_up + qid_p_up] == 1 && (empty_list[nid] == 0))
		{
			TN node = node_list[nid];

			dis_q = 0;
			if (data_info[2] == 2)
			{ // L2 distance
				for (int j = 0; j < data_info[0]; j++)
				{
					dis_q += pow(data_d[node.pid * data_info[0] + j] - data_d[qid_list[qid] * data_info[0] + j], 2);
				}
				dis_q = pow(dis_q, 0.5);
			}
			else if (data_info[2] == 1)
			{ // L1 distance
				for (int j = 0; j < data_info[0]; j++)
				{
					dis_q += abs(data_d[node.pid * data_info[0] + j] - data_d[qid_list[qid] * data_info[0] + j]);
				}
			}
			else if (data_info[2] == 0)
			{ // Max value
				float temp = 0;
				for (int j = 0; j < data_info[0]; j++)
				{
					temp = abs(data_d[node.pid * data_info[0] + j] - data_d[qid_list[qid] * data_info[0] + j]);
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
					sa2 += data_d[qid_list[qid] * data_info[0] + j] * data_d[qid_list[qid] * data_info[0] + j];
					sa3 += data_d[node.pid * data_info[0] + j] * data_d[qid_list[qid] * data_info[0] + j];
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
				int m = size_s[qid_list[qid]];
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
							int cost = (data_s[node.pid * M + j - 1] == data_s[qid_list[qid] * M + k - 1]) ? 0 : 1;
							table[j][k] = 1 + min(table[j - 1][k], table[j][k - 1]);
							table[j][k] = min(table[j - 1][k - 1] + cost, table[j][k]);
						}
					}
					dis_q = table[n][m];
				}
			}
		}

		// Save result.
		p_list_k[offset_p + i] = i;
		p_list_k[offset_p + ofst / 3 + i] = dis_q;
		p_list_k[offset_p + ofst / 3 * 2 + i] = double(dis_q / INFI_DIS + qid_p * DIS_CODE);
	}
}

// Update disk.
__global__ void updateDisK(int qnum_l, double *p_list_k, float *disk, int nnum_l, int offset_p, int qs, int k)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int total_num = gridDim.x * blockDim.x;
	int ofst = (nnum_l / TREE_ORDER * 3) * qnum_l; // Offset at current level.

	for (int i = tid; i < qnum_l; i += total_num)
	{
		int qid_p = i;		  // Query idx in p_list
		int qid = qid_p + qs; // Query idx in qid_lsit
		// float dis = INFI_DIS / INFI_DIS + qid_p * DIS_CODE;
		/*float* address = thrust::find(thrust::device, p_list_k + offset_p + ofst / 3 * 2 + nnum_l / TREE_ORDER * qid_p,
			p_list_k + offset_p + ofst / 3 * 2 + nnum_l / TREE_ORDER * (qid_p + 1), dis);
		int idx = address - (p_list_k + offset_p + ofst / 3 * 2 + nnum_l / TREE_ORDER * qid_p);*/
		int idx = p_list_k[offset_p + nnum_l / TREE_ORDER * qid_p + k - 1];

		if (disk[qid] > p_list_k[offset_p + ofst / 3 + idx])
		{
			disk[qid] = p_list_k[offset_p + ofst / 3 + idx];
			// printf("disk[qid]: %f, qid: %d\n", disk[qid], qid);
		}
	}
}

// Range query
void searchIndexRnnV2(short *data_d, TN *node_list, int *id_list, int *max_node_num, int *qid_list,
					  int qnum, float r, int tree_h, int *data_info, int *empty_list, char *data_s, int *size_s)
{
	cout << "Searching..." << endl;

	CHECK(cudaMallocManaged((void **)&res, qnum * sizeof(int)));
	CHECK(cudaMallocManaged((void **)&size_list, (tree_h + 1) * sizeof(int)));

	// Get GPU available memory.
	size_t avail;
	size_t total;
	cudaMemGetInfo(&avail, &total);
	// if (input_size <= 0 || input_size > avail) {
	// 	printf("Out of memory !!!\n");
	// 	return;
	// }
	// cout << "avail: " << avail << endl;
	// cout << "input: " << input_size << endl;
	// avail = input_size;
	avail = avail / 2; // Allocate storage space as a half of available space.
	// cout << "avail: " << avail << endl;
	// cout << "total: " << total << endl;

	// Allocate memory
	size_a = avail / sizeof(int); // Get the total int num.
	CHECK(cudaMalloc((void **)&p_list, size_a * sizeof(int)));
	// cout << "size_a: " << size_a << endl;

	// Initialize the query information
	CHECK(cudaMemset(size_list, 0, (tree_h + 1) * sizeof(int)));
	CHECK(cudaMemset(p_list, 0, size_a * sizeof(int)));
	size_list[0] = qnum;
	size_a -= qnum;
	size_avg = size_a / tree_h;
	nnum_l = TREE_ORDER;
	qnum_l = min(size_avg / nnum_l, qnum);
	size_a -= qnum_l * nnum_l;
	size_list[1] = qnum_l * nnum_l;
	for (int i = 0; i < qnum; i += qnum_l)
	{
		int end = min(i + qnum_l - 1, qnum - 1);
		st.push(i);
		st.push(end);
		st.push(1);
		st.push(qnum);
		st.push(1);
		st.push(0);
		st.push(size_a);
	}
	initPList<<<(qnum + THREAD_NUM - 1) / THREAD_NUM, THREAD_NUM>>>(p_list, qnum);
	cudaDeviceSynchronize();
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "initPlist error: %s\n", cudaGetErrorString(cudaStatus));
	/*initResV2 << < (qnum + THREAD_NUM - 1) / THREAD_NUM, THREAD_NUM >> > (res, qnum);
	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) fprintf(stderr, "initRes error: %s\n", cudaGetErrorString(cudaStatus));*/

	// Range query
	while (!st.empty())
	{
		// Get the preparation information for queries
		size_a = st.top();
		st.pop();
		qs_up = st.top();
		st.pop();
		offset_n = st.top();
		st.pop();
		qnum_up = st.top();
		st.pop();
		cur_level = st.top();
		st.pop();
		qe = st.top();
		st.pop();
		qs = st.top();
		st.pop();
		qnum_l = qe - qs + 1;
		offset_p = thrust::reduce(thrust::device, size_list, size_list + cur_level, 0);
		offset_up_p = offset_p - size_list[cur_level - 1];
		nnum_l = pow(TREE_ORDER, cur_level);
		int block_num = (qnum_l * nnum_l + THREAD_NUM - 1) / THREAD_NUM;

		// Evaluating
		if (cur_level < tree_h)
		{ // Processing node.
			nodeProcessRnn<<<block_num, THREAD_NUM>>>(node_list, r, data_d, qid_list, data_info, empty_list, data_s,
													  size_s, qnum_l, qnum_up, nnum_l, p_list, offset_p, offset_up_p, offset_n, qs, qs_up);
			cudaDeviceSynchronize();
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess)
				fprintf(stderr, "nodeProcessRnn error: %s\n", cudaGetErrorString(cudaStatus));
		}
		else
		{ // Processing data in leaf node.
			// Get query information.
			int ls = qs;
			int le = qe;
			int offset_up_n = offset_n;
			int nnum_up = pow(TREE_ORDER, cur_level - 1);

			// Get counts of query.
			block_num = (qnum_up * nnum_up + THREAD_NUM - 1) / THREAD_NUM;
			getQCount<<<block_num, THREAD_NUM>>>(ls, le, p_list, offset_up_p, offset_p, qnum_up, nnum_up);
			cudaDeviceSynchronize();
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess)
				fprintf(stderr, "getQCount error: %s\n", cudaGetErrorString(cudaStatus));

			// Gets the prefix sum of p_list at leaf node layer.
			int lnum = thrust::reduce(thrust::device, p_list + offset_p, p_list + offset_p + (le - ls) * nnum_up, 0);
			thrust::exclusive_scan(thrust::device, p_list + offset_p, p_list + offset_p + (le - ls) * nnum_up,
								   p_list + offset_p);

			// Merge leaf node.
			mergeLNode<<<block_num, THREAD_NUM>>>(ls, le, p_list, offset_up_p, offset_up_n, qs_up, offset_p, size_list,
												  cur_level, qnum_up, nnum_up);
			cudaDeviceSynchronize();
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess)
				fprintf(stderr, "mergeLNode error: %s\n", cudaGetErrorString(cudaStatus));

			// Processing data in leaf node.
			block_num = lnum;
			// printf("lnum: %d\n", block_num);
			dataProcessRnn<<<block_num, THREAD_NUM>>>(node_list, r, data_d, qid_list, data_info, data_s, size_s, p_list,
													  offset_p, id_list, cur_level, size_list, nnum_up);
			cudaDeviceSynchronize();
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess)
				fprintf(stderr, "dataProcessRnn error: %s\n", cudaGetErrorString(cudaStatus));

			// Merge result.
			block_num = (le - ls + THREAD_NUM - 1) / THREAD_NUM;
			mergeResRnn<<<block_num, THREAD_NUM>>>(ls, le, p_list, offset_p, size_list, cur_level, nnum_up, lnum, res);
			cudaDeviceSynchronize();
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess)
				fprintf(stderr, "mergeResRnn error: %s\n", cudaGetErrorString(cudaStatus));
		}

		// Update the query and storage space information and of lower layer.
		if (cur_level < tree_h)
		{
			int cur_level_low = cur_level + 1;
			int size_avg_low = size_a / (tree_h - cur_level);
			int offset_n_low = offset_n + nnum_l;

			if (cur_level_low < tree_h)
			{ // The lower layer evaluates the entire nodes.
				// Update the query and storage space information and of lower layer.
				int nnum_l_low = pow(TREE_ORDER, cur_level_low);
				int qnum_l_low = min(size_avg_low / nnum_l_low, size_list[cur_level] / nnum_l);
				size_list[cur_level_low] = qnum_l_low * nnum_l_low;
				for (int i = 0; i < qnum_l; i += qnum_l_low)
				{
					int end = min(i + qs + qnum_l_low - 1, qe);
					st.push(i + qs);
					st.push(end);
					st.push(cur_level_low);
					st.push(qnum_l);
					st.push(offset_n_low);
					st.push(qs);
					st.push(size_a - size_list[cur_level_low]);
				}
			}
			else
			{ // The lower layer evaluates the data in leaf nodes.
				// Update the query and storage space informationand of lower layer.
				unsigned long size_l = min((unsigned long)size_avg_low / (MAX_SIZE + 3), (unsigned long)size_list[cur_level]);
				size_list[cur_level_low] = size_l * (MAX_SIZE + 3);
				// printf("size_l: %d, MAX_SIZE: %d,  nnum_l: %d", size_l, MAX_SIZE, nnum_l);
				unsigned long qnum_l_low = size_l / (nnum_l);
				printf("qnum_l_low: %d\n", qnum_l_low);
				for (int i = 0; i < qnum_l; i += qnum_l_low)
				{
					int end = min((int)(i + qnum_l_low), qnum_l);
					st.push(i);
					st.push(end);
					st.push(cur_level_low);
					st.push(qnum_l);
					st.push(offset_n);
					st.push(qs);
					st.push(size_a - size_list[cur_level_low]);
				}
			}
		}
	}

	// Release memory
	cudaFree(p_list);
	cudaFree(size_list);
}

// knn query
void searchIndexKnnV2(short *data_d, TN *node_list, int *id_list, int *max_node_num, int *qid_list,
					  int qnum, int k, int tree_h, int *data_info, int *empty_list, char *data_s, int *size_s)
{
	cout << "Searching..." << endl;

	CHECK(cudaMallocManaged((void **)&res_dis, qnum * sizeof(float)));
	CHECK(cudaMallocManaged((void **)&size_list, (tree_h + 1) * sizeof(int)));
	CHECK(cudaMalloc((void **)&disk, qnum * sizeof(float)));

	// Get GPU available memory.
	size_t avail;
	size_t total;
	cudaMemGetInfo(&avail, &total);
	// if (input_size <= 0 || input_size > avail) {
	// 	printf("Out of memory !!!\n");
	// 	return;
	// }
	// cout << "avail: " << avail << endl;
	// cout << "input: " << input_size << endl;
	// avail = input_size;
	avail = avail / 2; // Allocate storage space as a half of available space.
	// cout << "avail: " << avail << endl;
	// cout << "total: " << total << endl;

	// Allocate memory
	size_a = avail / (sizeof(double)); // Get the total num.
	CHECK(cudaMalloc((void **)&p_list_k, size_a * sizeof(double)));
	// CHECK(cudaMalloc((void**)&p_list_dis, size_a * sizeof(float)));
	// CHECK(cudaMalloc((void**)&p_list_disc, size_a * sizeof(float)));
	// cout << "size_a: " << size_a << endl;

	// Initialize the query information
	CHECK(cudaMemset(size_list, 0, (tree_h + 1) * sizeof(int)));
	// CHECK(cudaMemset(p_list, 0, size_a * sizeof(int)));
	size_list[0] = qnum;
	size_a -= qnum;
	size_avg = size_a / tree_h;
	nnum_l = TREE_ORDER;
	qnum_l = min(size_avg / (nnum_l + nnum_l / TREE_ORDER * 3), qnum);
	size_a -= qnum_l * (nnum_l + nnum_l / TREE_ORDER);
	size_list[1] = qnum_l * (nnum_l + nnum_l / TREE_ORDER);
	for (int i = 0; i < qnum; i += qnum_l)
	{
		int end = min(i + qnum_l - 1, qnum - 1);
		st.push(i);
		st.push(end);
		st.push(1);
		st.push(qnum);
		st.push(1);
		st.push(0);
		st.push(size_a);
	}
	initPListKnn<<<(qnum + THREAD_NUM - 1) / THREAD_NUM, THREAD_NUM>>>(p_list_k, qnum);
	cudaDeviceSynchronize();
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "initPlist error: %s\n", cudaGetErrorString(cudaStatus));
	initDisK<<<(qnum + THREAD_NUM - 1) / THREAD_NUM, THREAD_NUM>>>(disk, qnum);
	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "initDisk error: %s\n", cudaGetErrorString(cudaStatus));

	// knn query
	while (!st.empty())
	{
		// Get the preparation information for queries
		size_a = st.top();
		st.pop();
		qs_up = st.top();
		st.pop();
		offset_n = st.top();
		st.pop();
		qnum_up = st.top();
		st.pop();
		cur_level = st.top();
		st.pop();
		qe = st.top();
		st.pop();
		qs = st.top();
		st.pop();
		qnum_l = qe - qs + 1;
		offset_p = thrust::reduce(thrust::device, size_list, size_list + cur_level, 0);
		offset_up_p = offset_p - size_list[cur_level - 1];
		nnum_l = pow(TREE_ORDER, cur_level);
		int block_num = (qnum_l * nnum_l + THREAD_NUM - 1) / THREAD_NUM;

		// Evaluating
		if (cur_level < tree_h)
		{ // Processing node.
			int pnum_level = nnum_l - thrust::reduce(thrust::device, empty_list + start_idx, empty_list + start_idx + nnum_l, 0);
			pnum_level = pnum_level / TREE_ORDER;
			// printf("pnum level: %d\n", pnum_level);
			int pnum_level_total = nnum_l / TREE_ORDER;
			// printf("pnum level total: %d\n", pnum_level_total);

			if (update_disk == false && (pnum_level < k || cur_level <= 2))
			{
				labelCNode<<<block_num, THREAD_NUM>>>(empty_list, qnum_l, qnum_up, nnum_l, p_list_k, offset_p, offset_up_p, offset_n, qs, qs_up);
				cudaDeviceSynchronize();
				cudaStatus = cudaGetLastError();
				if (cudaStatus != cudaSuccess)
					fprintf(stderr, "labelCNode error: %s\n", cudaGetErrorString(cudaStatus));
			}
			else
			{
				update_disk = true;

				if (pnum_level >= k)
				{
					// Compute the distances between pivots and queries at current level.
					block_num = (pnum_level_total * qnum_l + THREAD_NUM - 1) / THREAD_NUM;
					getDisPQ<<<block_num, THREAD_NUM>>>(node_list, data_d, qid_list, data_info, empty_list, data_s, size_s, qnum_l,
														qnum_up, nnum_l, p_list_k, offset_p, offset_up_p, offset_n, qs, qs_up, pnum_level_total);
					cudaDeviceSynchronize();
					cudaStatus = cudaGetLastError();
					if (cudaStatus != cudaSuccess)
						fprintf(stderr, "getDisPQ error: %s\n", cudaGetErrorString(cudaStatus));

					// Sort by distances.
					int ofst = (nnum_l / TREE_ORDER * 3) * qnum_l; // Offset at current level.
					thrust::sort_by_key(thrust::device, p_list_k + offset_p + ofst / 3 * 2,
										p_list_k + offset_p + ofst / 3 * 2 + pnum_level_total * qnum_l, p_list_k + offset_p);
					cudaStatus = cudaGetLastError();
					if (cudaStatus != cudaSuccess)
						fprintf(stderr, "sort_by_key error: %s\n", cudaGetErrorString(cudaStatus));

					// Update disk.
					updateDisK<<<(qnum_l + THREAD_NUM - 1) / THREAD_NUM, THREAD_NUM>>>(qnum_l, p_list_k, disk, nnum_l, offset_p, qs, k);
					cudaDeviceSynchronize();
					cudaStatus = cudaGetLastError();
					if (cudaStatus != cudaSuccess)
						fprintf(stderr, "updateDisK error: %s\n", cudaGetErrorString(cudaStatus));
				}

				// Process the nodes of the current layer and determine if the node will be pruned.
				block_num = (nnum_l * qnum_l + THREAD_NUM - 1) / THREAD_NUM;
				nodeProcessKnn<<<block_num, THREAD_NUM>>>(node_list, disk, empty_list, qnum_l, qnum_up, nnum_l, p_list_k, offset_p,
														  offset_up_p, offset_n, qs, qs_up);
				cudaDeviceSynchronize();
				cudaStatus = cudaGetLastError();
				if (cudaStatus != cudaSuccess)
					fprintf(stderr, "nodeProcessKnn error: %s\n", cudaGetErrorString(cudaStatus));
			}
		}
		else
		{ // Processing data in leaf node.
			// Get query information.
			int ls = qs;
			int le = qe;
			int offset_up_n = offset_n;
			int nnum_up = pow(TREE_ORDER, cur_level - 1);

			// Get counts of query.
			block_num = (qnum_up * nnum_up + THREAD_NUM - 1) / THREAD_NUM;
			getQCountKnn<<<block_num, THREAD_NUM>>>(ls, le, p_list_k, offset_up_p, offset_p, qnum_up, nnum_up);
			cudaDeviceSynchronize();
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess)
				fprintf(stderr, "getQCountKnn error: %s\n", cudaGetErrorString(cudaStatus));

			// Gets the prefix sum of p_list at leaf node layer.
			int lnum = thrust::reduce(thrust::device, p_list_k + offset_p, p_list_k + offset_p + (le - ls) * nnum_up, 0);
			thrust::exclusive_scan(thrust::device, p_list_k + offset_p, p_list_k + offset_p + (le - ls) * nnum_up,
								   p_list_k + offset_p);
			// printf("lnum: %d\n", lnum);

			// Merge leaf node.
			mergeLNodeKnn<<<block_num, THREAD_NUM>>>(ls, le, p_list_k, offset_up_p, offset_up_n, qs_up, offset_p, size_list,
													 cur_level, qnum_up, nnum_up);
			cudaDeviceSynchronize();
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess)
				fprintf(stderr, "mergeLNodeKnn error: %s\n", cudaGetErrorString(cudaStatus));

			// Processing data in leaf node.
			block_num = lnum;
			dataProcessKnn<<<block_num, THREAD_NUM>>>(node_list, disk, data_d, qid_list, data_info, data_s, size_s, p_list_k,
													  offset_p, id_list, cur_level, size_list, nnum_up);
			cudaDeviceSynchronize();
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess)
				fprintf(stderr, "dataProcessKnn error: %s\n", cudaGetErrorString(cudaStatus));

			// Sort by distances.
			thrust::sort_by_key(thrust::device, p_list_k + offset_p + size_list[cur_level] / (MAX_SIZE * 3 + 3) * (3 + 2 * MAX_SIZE),
								p_list_k + offset_p + size_list[cur_level] / (MAX_SIZE * 3 + 3) * (3 + 2 * MAX_SIZE) + lnum * MAX_SIZE,
								p_list_k + offset_p + size_list[cur_level] / (MAX_SIZE * 3 + 3) * 3);
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess)
				fprintf(stderr, "sort_by_key error: %s\n", cudaGetErrorString(cudaStatus));

			// Merge result.
			block_num = (le - ls + THREAD_NUM - 1) / THREAD_NUM;
			mergeResKnn<<<block_num, THREAD_NUM>>>(ls, le, p_list_k, offset_p, size_list, cur_level, nnum_up, res_dis, k);
			cudaDeviceSynchronize();
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess)
				fprintf(stderr, "mergeResKnn error: %s\n", cudaGetErrorString(cudaStatus));
		}

		// Update the query and storage space information and of lower layer.
		if (cur_level < tree_h)
		{
			int cur_level_low = cur_level + 1;
			int size_avg_low = size_a / (tree_h - cur_level);
			int offset_n_low = offset_n + nnum_l;

			if (cur_level_low < tree_h)
			{ // The lower layer evaluates the entire nodes.
				// Update the query and storage space information and of lower layer.
				int nnum_l_low = pow(TREE_ORDER, cur_level_low);
				int qnum_l_low = min(size_avg_low / (nnum_l_low + nnum_l_low / TREE_ORDER * 3), qnum_up);
				size_list[cur_level_low] = qnum_l_low * (nnum_l_low + nnum_l_low / TREE_ORDER * 3);
				for (int i = 0; i < qnum_l; i += qnum_l_low)
				{
					int end = min(i + qs + qnum_l_low - 1, qe);
					st.push(i + qs);
					st.push(end);
					st.push(cur_level_low);
					st.push(qnum_l);
					st.push(offset_n_low);
					st.push(qs);
					st.push(size_a - size_list[cur_level_low]);
				}
			}
			else
			{ // The lower layer evaluates the data in leaf nodes.
				// Update the query and storage space informationand of lower layer.
				unsigned long size_l = min((unsigned long)size_avg_low / (MAX_SIZE * 3 + 3), (unsigned long)qnum_l * nnum_l);
				size_list[cur_level_low] = size_l * (MAX_SIZE * 3 + 3);
				// printf("size_l: %d, MAX_SIZE: %d,  nnum_l: %d\n", size_l, MAX_SIZE, nnum_l);
				unsigned long qnum_l_low = size_l / (nnum_l);
				printf("qnum_l_low: %d\n", qnum_l_low);
				for (int i = 0; i < qnum_l; i += qnum_l_low)
				{
					int end = min((int)(i + qnum_l_low), qnum_l);
					st.push(i);
					st.push(end);
					st.push(cur_level_low);
					st.push(qnum_l);
					st.push(offset_n);
					st.push(qs);
					st.push(size_a - size_list[cur_level_low]);
				}
			}
		}
	}

	// Release memory
	cudaFree(p_list_k);
	cudaFree(size_list);
	cudaFree(disk);
}