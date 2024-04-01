// GTS update
// Created on 24-01-05

#pragma once
#include <chrono>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <stdio.h>
#include <thrust/reduce.h>
#include "search.cuh"
#include "tree.cuh"
#include "search_naive.cuh"
#include "config.cuh"

__managed__ int MAX_IN_SIZE = 10;

typedef struct UO
{
	int update_flag;
	int update_id;
};
int update_num;
UO *update_list;
int *insert_list;
int *insert_list_temp;
int *is_delete;
int *is_delete_prefix;
int in_size;
int tree_size;
Obj obj_r;
int rnum[1];
int total_result_num;
int *total_result_id;
float *total_result_dis;
int *delete_id;
int *is_delete_in;
int *is_delete_in_prefix;
short *data_d_temp;
char *data_s_temp;
int *size_s_temp;

void loadUpdate(char *file, UO *&update_list, int &update_num)
{
	ifstream in(file);
	if (!in.is_open())
	{
		std::cout << "open file error" << std::endl;
		exit(-1);
	}

	cout << "Loading update file..." << endl;

	string line;
	int i = 0;
	int j = 0;
	vector<string> res;

	// load the file
	while (getline(in, line))
	{
		if (i == 0)
		{ // load the first line
			stringstream ss(line);
			int number;
			ss >> number;

			cudaMallocManaged((void **)&update_list, number * sizeof(UO));
			update_num = number;
		}
		else
		{ // load update object
			split(line, res, ' ');
			for (auto r : res)
			{
				stringstream ss(r);
				int number;
				ss >> number;

				if (j == 0)
					update_list[i - 1].update_flag = number;
				if (j == 1)
					update_list[i - 1].update_id = number;

				j++;
			}
		}

		res.clear();
		j = 0;
		i++;
	}

	in.close();
}

__global__ void mergeTotalResult(int total_result_num, int *total_result_id, int *qresult_count, int *result_id, float *result_dis,
								 Obj obj_r, float *total_result_dis, int *is_delete_prefix, int tree_size)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int total_num = gridDim.x * blockDim.x;

	for (int idx = id; idx < total_result_num; idx += total_num)
	{
		if (idx < qresult_count[0])
		{
			total_result_id[idx] = result_id[idx] - is_delete_prefix[result_id[idx]];
			total_result_dis[idx] = result_dis[idx];
		}
		else
		{
			total_result_id[idx] = obj_r.res_id_q[idx - qresult_count[0]] + tree_size - is_delete_prefix[tree_size - 1];
			total_result_dis[idx] = obj_r.dis_q[idx - qresult_count[0]];
		}
	}
}

__global__ void findIdx(int *id_cur, int id_u, int *is_delete_prefix, int tree_size, int in_size, int *is_delete)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int total_num = gridDim.x * blockDim.x;
	int num = tree_size + in_size;

	for (int idx = id; idx < num; idx += total_num)
	{
		int data_id = -1;

		if (idx < tree_size)
		{
			if (is_delete[idx] == 0)
			{
				data_id = idx - is_delete_prefix[idx];
			}
		}
		else
		{
			data_id = idx - is_delete_prefix[tree_size - 1];
		}

		if (data_id == id_u)
			id_cur[0] = idx;
	}
}

__global__ void mergeInResult(int in_size, int *insert_list, int *is_delete_in, int *insert_list_temp, int *is_delete_in_prefix)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int total_num = gridDim.x * blockDim.x;

	for (int idx = id; idx < in_size; idx += total_num)
	{
		if (is_delete_in[idx] == 0)
		{
			insert_list[idx - is_delete_in_prefix[idx]] = insert_list_temp[idx];
		}
	}
}

__global__ void getNewData(short *data_d, short *data_d_temp, char *data_s, char *data_s_temp, int *size_s, int *size_s_temp,
						   int *is_delete, int *is_delete_prefix, int in_size, int *insert_list, int *data_info, int tree_size)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int total_num = gridDim.x * blockDim.x;

	for (int idx = id; idx < (in_size + tree_size); idx += total_num)
	{
		if (data_info[2] != 6)
		{
			if (idx < tree_size)
			{
				if (is_delete[idx] == 0)
				{
					int i = idx - is_delete_prefix[idx];
					for (int j = 0; j < data_info[0]; j++)
					{
						data_d[i * data_info[0] + j] = data_d_temp[idx * data_info[0] + j];
					}
				}
			}
			else
			{
				int i = idx - is_delete_prefix[tree_size - 1];
				int i_in = idx - tree_size;
				for (int j = 0; j < data_info[0]; j++)
				{
					data_d[i * data_info[0] + j] = data_d_temp[insert_list[i_in] * data_info[0] + j];
				}
			}
		}

		else
		{
			if (idx < tree_size)
			{
				if (is_delete[idx] == 0)
				{
					int i = idx - is_delete_prefix[idx];
					for (int j = 0; j < size_s_temp[idx]; j++)
					{
						data_s[i * M + j] = data_s_temp[idx * M + j];
					}
					size_s[i] = size_s_temp[idx];
				}
			}
			else
			{
				int i = idx - is_delete_prefix[tree_size - 1];
				int i_in = idx - tree_size;
				for (int j = 0; j < size_s_temp[insert_list[i_in]]; j++)
				{
					data_s[i * M + j] = data_s_temp[insert_list[i_in] * M + j];
				}
				size_s[i] = size_s_temp[insert_list[i_in]];
			}
		}
	}
}

__global__ void leafProcessRnnUpdate(int *query_lnode, TN *node_list, int *id_list, int *query_qid, short *data_d,
									 int *qid_list, int *init_result_id, float *init_result_dis, int *data_info, float r, int *qresult_idx,
									 int *search_num, char *data_s, int *size_s, int *is_delete)
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

			if (is_delete[data_id] == 0)
			{
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
}

void searchIndexRnnUpdate(short *data_d, TN *node_list, int *id_list, int *max_node_num, int *qid_list,
						  int qnum, float r, int tree_h, int *data_info, int *&empty_list, int *&qresult_count,
						  int *&qresult_count_prefix, int *&result_id, float *&result_dis, char *data_s, int *size_s)
{
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

	for (int i = 0; i < qnum; i = i + qnum_leaf)
	{
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

		leafProcessRnnUpdate<<<search_num[0], THREAD_NUM>>>(query_lnode, node_list, id_list, query_qid, data_d, qid_list,
															init_result_id, init_result_dis, data_info, r, qresult_idx, search_num, data_s, size_s, is_delete);
		cudaDeviceSynchronize();
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
			fprintf(stderr, "leafProcess error: %s\n", cudaGetErrorString(cudaStatus));

		result_num[0] = thrust::reduce(thrust::device, qresult_idx, qresult_idx + (search_num[0] * MAX_SIZE), 0);
		// printf("result num: %d\n", result_num[0]);
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

		cudaFree(init_result_id);
		cudaFree(init_result_dis);
		cudaFree(qresult_idx);
		cudaFree(query_lnode);
		cudaFree(query_qid);
	}

	cudaFree(query_node_list);
	cudaFree(search_num);
	cudaFree(result_num);
	cudaFree(qnode_count);
	cudaFree(qnode_count_prefix);
	cudaFree(qnode_idx);
}

void updateIndexRnn(short *&data_d, TN *&node_list, int *&id_list, int *&max_node_num, int *&qid_list, int qnum, float r, int &tree_h,
					int *&data_info, int *&empty_list, int *&qresult_count, int *&qresult_count_prefix, int *&result_id, float *&result_dis,
					char *&data_s, int *&size_s, FILE *fcost, float &time_update_s, float &time_update_u, int &count_update_s, int &count_update_u)
{
	printf("Updating...\n");

	auto s = std::chrono::high_resolution_clock::now();
	CHECK(cudaMallocManaged((void **)&is_delete_in, MAX_IN_SIZE * sizeof(int)));
	CHECK(cudaMallocManaged((void **)&is_delete, data_info[1] * sizeof(int)));
	CHECK(cudaMallocManaged((void **)&qid_list, sizeof(int)));
	CHECK(cudaMallocManaged((void **)&delete_id, sizeof(int)));
	CHECK(cudaMallocManaged((void **)&insert_list, MAX_IN_SIZE * sizeof(int)));
	CHECK(cudaMalloc((void **)&insert_list_temp, MAX_IN_SIZE * sizeof(int)));
	CHECK(cudaMallocManaged((void **)&is_delete_prefix, data_info[1] * sizeof(int)));
	CHECK(cudaMalloc((void **)&is_delete_in_prefix, MAX_IN_SIZE * sizeof(int)));
	CHECK(cudaMallocManaged((void **)&obj_r.dis_q, MAX_IN_SIZE * sizeof(float)));
	CHECK(cudaMallocManaged((void **)&obj_r.res_id_q, MAX_IN_SIZE * sizeof(int)));
	CHECK(cudaMemset(is_delete, 0, data_info[1] * sizeof(int)));
	in_size = 0;
	tree_size = data_info[1];
	auto e = std::chrono::high_resolution_clock::now();
	std::chrono::duration<float> diff = e - s;
	time_update_u += diff.count();

	for (int i = 0; i < update_num; i++)
	{
		if (update_list[i].update_flag == 0)
		{
			count_update_u++;
			s = std::chrono::high_resolution_clock::now();
			// printf("Inserting ...\n");

			insert_list[in_size] = update_list[i].update_id;
			in_size++;

			if (in_size == MAX_IN_SIZE)
			{
				// printf("%d\n", data_info[1]);
				if (data_info[2] != 6)
				{
					CHECK(cudaMalloc((void **)&data_d_temp, data_info[1] * data_info[0] * sizeof(short)));
					CHECK(cudaMemcpy(data_d_temp, data_d, data_info[1] * data_info[0] * sizeof(short), cudaMemcpyDeviceToDevice));
					cudaFree(data_d);
					thrust::inclusive_scan(thrust::device, is_delete, is_delete + tree_size, is_delete_prefix);
					data_info[1] = tree_size - is_delete_prefix[tree_size - 1] + in_size;
					CHECK(cudaMallocManaged((void **)&data_d, data_info[1] * data_info[0] * sizeof(short)));
				}
				else
				{
					CHECK(cudaMalloc((void **)&data_s_temp, data_info[1] * M * sizeof(char)));
					CHECK(cudaMalloc((void **)&size_s_temp, data_info[1] * sizeof(int)));
					CHECK(cudaMemcpy(data_s_temp, data_s, data_info[1] * M * sizeof(char), cudaMemcpyDeviceToDevice));
					CHECK(cudaMemcpy(size_s_temp, size_s, data_info[1] * sizeof(int), cudaMemcpyDeviceToDevice));
					cudaFree(data_s);
					cudaFree(size_s);
					thrust::inclusive_scan(thrust::device, is_delete, is_delete + tree_size, is_delete_prefix);
					data_info[1] = tree_size - is_delete_prefix[tree_size - 1] + in_size;
					CHECK(cudaMallocManaged((void **)&data_s, data_info[1] * M * sizeof(char)));
					CHECK(cudaMallocManaged((void **)&size_s, data_info[1] * sizeof(int)));
				}
				// printf("%d\n", data_info[1]);

				getNewData<<<(in_size + tree_size - 1) / THREAD_NUM + 1, THREAD_NUM>>>(data_d, data_d_temp, data_s, data_s_temp, size_s,
																					   size_s_temp, is_delete, is_delete_prefix, in_size, insert_list, data_info, tree_size);
				cudaDeviceSynchronize();
				cudaError_t cudaStatus = cudaGetLastError();
				if (cudaStatus != cudaSuccess)
					fprintf(stderr, "getNewData error: %s\n", cudaGetErrorString(cudaStatus));

				CHECK(cudaFree(max_node_num));
				CHECK(cudaFree(empty_list));
				CHECK(cudaFree(id_list));
				CHECK(cudaFree(node_list));
				if (data_info[2] != 6)
				{
					cudaFree(data_d_temp);
				}
				else
				{
					CHECK(cudaFree(data_s_temp));
					CHECK(cudaFree(size_s_temp));
				}
				indexConstru(data_d, data_s, size_s, data_info, id_list, node_list, max_node_num, tree_h, empty_list);

				tree_size = data_info[1];
				CHECK(cudaFree(is_delete));
				CHECK(cudaFree(is_delete_prefix));
				CHECK(cudaMallocManaged((void **)&is_delete, data_info[1] * sizeof(int)));
				CHECK(cudaMallocManaged((void **)&is_delete_prefix, data_info[1] * sizeof(int)));
				CHECK(cudaMemset(is_delete, 0, data_info[1] * sizeof(int)));
				in_size = 0;
			}
			e = std::chrono::high_resolution_clock::now();
			diff = e - s;
			time_update_u += diff.count();
		}

		else if (update_list[i].update_flag == 1)
		{
			count_update_u++;
			s = std::chrono::high_resolution_clock::now();
			// printf("Deleting ...\n");

			thrust::inclusive_scan(thrust::device, is_delete, is_delete + tree_size, is_delete_prefix);
			findIdx<<<(tree_size + in_size - 1) / THREAD_NUM + 1, THREAD_NUM>>>(delete_id, update_list[i].update_id,
																				is_delete_prefix, tree_size, in_size, is_delete);
			cudaDeviceSynchronize();
			cudaError_t cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess)
				fprintf(stderr, "findIdx error: %s\n", cudaGetErrorString(cudaStatus));
			// printf("%d\n", delete_id[0]);

			if (delete_id[0] < tree_size)
			{
				is_delete[delete_id[0]] = 1;
			}
			else
			{
				CHECK(cudaMemset(is_delete_in, 0, in_size * sizeof(int)));
				CHECK(cudaMemcpy(insert_list_temp, insert_list, in_size * sizeof(int), cudaMemcpyDeviceToDevice));
				is_delete_in[delete_id[0] - tree_size] = 1;
				thrust::inclusive_scan(thrust::device, is_delete_in, is_delete_in + in_size, is_delete_in_prefix);
				mergeInResult<<<(in_size - 1) / THREAD_NUM + 1, THREAD_NUM>>>(in_size, insert_list, is_delete_in,
																			  insert_list_temp, is_delete_in_prefix);
				cudaDeviceSynchronize();
				cudaStatus = cudaGetLastError();
				if (cudaStatus != cudaSuccess)
					fprintf(stderr, "mergeInResult error: %s\n", cudaGetErrorString(cudaStatus));
				in_size--;
			}
			e = std::chrono::high_resolution_clock::now();
			diff = e - s;
			time_update_u += diff.count();
		}

		else
		{
			count_update_s++;
			s = std::chrono::high_resolution_clock::now();
			qid_list[0] = update_list[i].update_id;

			searchIndexRnnUpdate(data_d, node_list, id_list, max_node_num, qid_list, qnum, r, tree_h, data_info, empty_list,
								 qresult_count, qresult_count_prefix, result_id, result_dis, data_s, size_s);
			if (in_size > 0)
			{
				searchNaiveRnn(data_info, obj_r, data_d, data_s, size_s, qid_list[0], in_size, r, insert_list, rnum);
			}

			total_result_num = qresult_count[0] + rnum[0];
			// printf("total result num: %d\n", total_result_num);
			CHECK(cudaMallocManaged((void **)&total_result_id, total_result_num * sizeof(int)));
			CHECK(cudaMallocManaged((void **)&total_result_dis, total_result_num * sizeof(float)));
			thrust::inclusive_scan(thrust::device, is_delete, is_delete + tree_size, is_delete_prefix);
			mergeTotalResult<<<(total_result_num - 1) / THREAD_NUM + 1, THREAD_NUM>>>(total_result_num, total_result_id, qresult_count,
																					  result_id, result_dis, obj_r, total_result_dis, is_delete_prefix, tree_size);
			cudaDeviceSynchronize();
			cudaError_t cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess)
				fprintf(stderr, "mergeTotalResult error: %s\n", cudaGetErrorString(cudaStatus));

			fprintf(fcost, "%d ", total_result_num);
			fflush(fcost);

			CHECK(cudaFree(total_result_id));
			CHECK(cudaFree(total_result_dis));
			CHECK(cudaFree(qresult_count));
			CHECK(cudaFree(qresult_count_prefix));
			CHECK(cudaFree(result_id));
			CHECK(cudaFree(result_dis));

			e = std::chrono::high_resolution_clock::now();
			diff = e - s;
			time_update_s += diff.count();
		}
	}

	s = std::chrono::high_resolution_clock::now();
	CHECK(cudaFree(obj_r.dis_q));
	CHECK(cudaFree(obj_r.res_id_q));
	cudaFree(insert_list);
	cudaFree(insert_list_temp);
	cudaFree(is_delete);
	cudaFree(is_delete_prefix);
	cudaFree(delete_id);
	cudaFree(is_delete_in);
	cudaFree(is_delete_in_prefix);
	e = std::chrono::high_resolution_clock::now();
	diff = e - s;
	time_update_u += diff.count();
}
