// Search with GPU-B+tree
// Created on 24-01-05

#pragma once
#include "priority_queue.cuh"
#include "config.cuh"

// Compute the distance between pivots and query points
// Block -> query
// Thread -> pivot of each partition, compute the distance between pivot and query
__global__ void getPivotDis(short *data_d, int *pid, int *qid, int qnum, float *dis_pivot, int *data_info, char *data_s, int *size_s)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;

	if (bid < qnum)
	{
		for (int id = tid; id < PNUM; id += blockDim.x)
		{
			float result = 0;

			if (data_info[2] == 2)
			{ // L2 distance
				for (int i = 0; i < data_info[0]; i++)
				{
					result += pow(data_d[pid[id] * data_info[0] + i] - data_d[qid[bid] * data_info[0] + i], 2);
				}
				result = pow(result, 0.5);
			}
			else if (data_info[2] == 1)
			{ // L1 distance
				for (int i = 0; i < data_info[0]; i++)
				{
					result += abs(data_d[pid[id] * data_info[0] + i] - data_d[qid[bid] * data_info[0] + i]);
				}
			}
			else if (data_info[2] == 0)
			{ // Max value
				float temp = 0;
				for (int i = 0; i < data_info[0]; i++)
				{
					temp = abs(data_d[pid[id] * data_info[0] + i] - data_d[qid[bid] * data_info[0] + i]);
					if (temp > result)
						result = temp;
				}
			}
			else if (data_info[2] == 5)
			{
				float sa1 = 0, sa2 = 0, sa3 = 0;
				for (int i = 0; i < data_info[0]; i++)
				{
					sa1 += data_d[pid[id] * data_info[0] + i] * data_d[pid[id] * data_info[0] + i];
					sa2 += data_d[qid[bid] * data_info[0] + i] * data_d[qid[bid] * data_info[0] + i];
					sa3 += data_d[pid[id] * data_info[0] + i] * data_d[qid[bid] * data_info[0] + i];
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
				int n = size_s[pid[id]];
				int m = size_s[qid[bid]];
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
							int cost = (data_s[pid[id] * MaxC + i - 1] == data_s[qid[bid] * MaxC + j - 1]) ? 0 : 1;
							table[i][j] = 1 + min(table[i - 1][j], table[i][j - 1]);
							table[i][j] = min(table[i - 1][j - 1] + cost, table[i][j]);
						}
					}
					result = table[n][m];
				}
			}

			int idx = bid * PNUM + id;
			dis_pivot[idx] = result;
		}
	}
}

// Check whether the partition can be pruned
// Block -> query
// Thread -> pivot of each partition, compute the distance between pivot and query
__global__ void pivotFilterRnn(float *dis_pivot, int qnum, float r, float *radius, int *isSatisfied, int *tree_num)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;

	if (bid < qnum)
	{
		for (int id = tid; id < PNUM; id += blockDim.x)
		{
			int idx = bid * PNUM + id;
			float dis_lb = dis_pivot[idx] - radius[id];
			if (dis_lb > r)
			{
				isSatisfied[idx] = 0;
			}
			else
			{
				isSatisfied[idx] = tree_num[id];
			}
		}
	}
}

// Check whether the partition can be pruned
// Block -> query
// Thread -> pivot of each partition, compute the distance between pivot and query
__global__ void pivotFilterKnn(int qnum, float *dis_ub, float *dis_lb, int *isSatisfied, int *tree_num, int k)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;

	if (bid < qnum)
	{
		float r = dis_ub[bid * PNUM + k - 1];

		for (int id = tid; id < PNUM; id += blockDim.x)
		{
			int idx = bid * PNUM + id;
			if (dis_lb[idx] > r)
			{
				isSatisfied[idx] = 0;
			}
			else
			{
				isSatisfied[idx] = tree_num[id];
			}
		}
	}
}

// Check whether the tree can be pruned
// Block -> query
// Thread -> B+-tree
__global__ void treeFilterKnn(int *root_idx, BPlusNode **T, int *qnum_counter, int *qnum_counter_prefix, int qnum, float *dis_pivot,
							  int *pivot_flag, int *qid, float *knn_bound, int *tree_filter)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;

	if (bid < qnum)
	{
		for (int id = tid; id < qnum_counter[bid]; id += blockDim.x)
		{
			int idx = qnum_counter_prefix[bid] + id;
			BPlusNode *node_temp;
			node_temp = T[root_idx[idx]];

			float dis = node_temp->Key[0] - dis_pivot[pivot_flag[idx] + bid * PNUM];
			dis = max(dis, 0.0);

			if (dis > knn_bound[bid])
			{
				tree_filter[idx] = 0;
				atomicAdd(&qnum_counter[bid], -1);
			}
			else
			{
				tree_filter[idx] = 1;
			}
		}
	}
}

// Check whether the tree can be pruned
// Block -> query
// Thread -> B+-tree
__global__ void treeFilterRnn(int *root_idx, BPlusNode **T, int *qnum_counter, int *qnum_counter_prefix, int qnum, float *dis_pivot,
							  int *pivot_flag, int *qid, float r, int *tree_filter)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;

	if (bid < qnum)
	{
		for (int id = tid; id < qnum_counter[bid]; id += blockDim.x)
		{
			int idx = qnum_counter_prefix[bid] + id;
			BPlusNode *node_temp;
			node_temp = T[root_idx[idx]];

			float dis = node_temp->Key[0] - dis_pivot[pivot_flag[idx] + bid * PNUM];
			dis = max(dis, 0.0);

			if (dis > r)
			{
				tree_filter[idx] = 0;
				atomicAdd(&qnum_counter[bid], -1);
			}
			else
			{
				tree_filter[idx] = 1;
			}
		}
	}
}

__global__ void mergeTree(int *tree_filter_prefix, int *tree_filter, int total_c, int *root_idx, int *root_idx_f, int *pivot_flag, int *pivot_flag_f)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int total_num = gridDim.x * blockDim.x;

	for (int idx = id; idx < total_c; idx += total_num)
	{
		if (tree_filter[idx] > 0)
		{
			root_idx_f[tree_filter_prefix[idx]] = root_idx[idx];
			pivot_flag_f[tree_filter_prefix[idx]] = pivot_flag[idx];
		}
	}
}

// Check whether the partition can be pruned
// Block -> query
// Thread -> B+-tree for each query
__global__ void pivotFilterKnnAgain(int qnum, float *knn_bound, float *dis_lb, int *isSatisfied, int *tree_num)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;

	if (bid < qnum)
	{
		float r = knn_bound[bid];

		for (int id = tid; id < PNUM; id += blockDim.x)
		{
			int idx = bid * PNUM + id;
			if (dis_lb[idx] > r)
			{
				isSatisfied[idx] = 0;
			}
			else
			{
				isSatisfied[idx] = tree_num[id];
			}
		}
	}
}

// Range search
// Block -> query
// Thread -> Search B+-tree for each query
__global__ void searchRnnD(int *root_idx, BPlusNode **T, int *qnum_counter, int *qnum_counter_prefix, int qnum,
						   SeqQueue **SQ, float *dis_pivot, int *pivot_flag, float r, short *data_d, int *data_info, int *qid, Obj init_result,
						   int *result_counter, char *data_s, int *size_s)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;

	if (bid < qnum)
	{
		for (int id = tid; id < qnum_counter[bid]; id += blockDim.x)
		{
			// if (tid == 0) {
			//	int id = tid;
			int idx = qnum_counter_prefix[bid] + id;
			BPlusNode *tree_id;
			result_counter[idx] = 0;

			// Search
			InitQueue(SQ[idx]);
			EnterQueue(SQ[idx], T[root_idx[idx]]);

			while (!IsEmpty(SQ[idx]))
			{
				DeleteQueeue(SQ[idx], tree_id);
				for (int i = 0; i < tree_id->KeyNum[0]; i++)
				{
					float dis_lb = tree_id->Key[i] - dis_pivot[pivot_flag[idx] + bid * PNUM];
					if (dis_lb > r)
					{
						// printf("break\n");
						break;
					}

					if (i != tree_id->KeyNum[0] - 1)
					{
						dis_lb = dis_pivot[pivot_flag[idx] + bid * PNUM] - tree_id->Key[i + 1];
					}
					if (dis_lb > r)
					{
						// printf("continue\n");
						continue;
					}

					if (tree_id->Children[i] != NULL)
					{
						// int temp_id = tree_id->Children[i]->idx[0];
						EnterQueue(SQ[idx], tree_id->Children[i]);
					}
					else if (tree_id->Children[i] == NULL)
					{
						int data_id = tree_id->id[i];
						float result = 0;

						// Compute the real distance
						if (data_info[2] == 2)
						{ // L2 distance
							for (int i = 0; i < data_info[0]; i++)
							{
								result += pow(data_d[data_id * data_info[0] + i] - data_d[qid[bid] * data_info[0] + i], 2);
							}
							result = pow(result, 0.5);
						}
						else if (data_info[2] == 1)
						{ // L1 distance
							for (int i = 0; i < data_info[0]; i++)
							{
								result += abs(data_d[data_id * data_info[0] + i] - data_d[qid[bid] * data_info[0] + i]);
							}
						}
						else if (data_info[2] == 0)
						{ // Max value
							float temp = 0;
							for (int i = 0; i < data_info[0]; i++)
							{
								temp = abs(data_d[data_id * data_info[0] + i] - data_d[qid[bid] * data_info[0] + i]);
								if (temp > result)
									result = temp;
							}
						}
						else if (data_info[2] == 5)
						{
							float sa1 = 0, sa2 = 0, sa3 = 0;
							for (int i = 0; i < data_info[0]; i++)
							{
								sa1 += data_d[data_id * data_info[0] + i] * data_d[data_id * data_info[0] + i];
								sa2 += data_d[qid[bid] * data_info[0] + i] * data_d[qid[bid] * data_info[0] + i];
								sa3 += data_d[data_id * data_info[0] + i] * data_d[qid[bid] * data_info[0] + i];
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
							int m = size_s[qid[bid]];
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
										int cost = (data_s[data_id * MaxC + i - 1] == data_s[qid[bid] * MaxC + j - 1]) ? 0 : 1;
										table[i][j] = 1 + min(table[i - 1][j], table[i][j - 1]);
										table[i][j] = min(table[i - 1][j - 1] + cost, table[i][j]);
									}
								}
								result = table[n][m];
							}
						}

						// Save result
						if (result <= r)
						{
							int res_idx = result_counter[idx] + idx * DNUM;
							init_result.dis[res_idx] = result;
							init_result.res_id[res_idx] = data_id;
							result_counter[idx]++;
						}
					}
				}
			}
		}
	}
}

// knn search
// Block -> query
// Thread -> Search B+-tree for each query
__global__ void searchKnnD(int *root_idx, BPlusNode **T, int *qnum_counter, int *qnum_counter_prefix, int qnum, float *dis_pivot,
						   int *pivot_flag, short *data_d, int *data_info, int *qid, int k, PriorityQueue_k **pq_a, PriorityQueue **pq_c, float *knn_bound,
						   char *data_s, int *size_s)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;

	if (bid < qnum)
	{
		for (int id = tid; id < qnum_counter[bid]; id += blockDim.x)
		{
			int idx = qnum_counter_prefix[bid] + id;
			float dis_k = knn_bound[bid];
			// float dis_k = 999999;
			float dis_temp;
			int id_temp;
			BPlusNode *node_temp;
			float dis_res;
			int id_res;
			BPlusNode *node_res;

			initPQK(k, pq_a[idx], 0);
			initPQ(DNUM, pq_c[idx], 1);

			dis_temp = 0;
			id_temp = -1;
			node_temp = T[root_idx[idx]];

			pushPQ(node_temp, dis_temp, id_temp, pq_c[idx]);

			while (!isEmptyPQ(pq_c[idx]))
			{
				popPQ(pq_c[idx], node_temp, dis_temp, id_temp);

				if (dis_temp > dis_k)
					continue;

				for (int i = 0; i < node_temp->KeyNum[0]; i++)
				{
					if (node_temp->Children[i] != NULL)
					{
						float dis = node_temp->Key[i] - dis_pivot[pivot_flag[idx] + bid * PNUM];
						if (i != node_temp->KeyNum[0] - 1)
						{
							float dis_lb = dis_pivot[pivot_flag[idx] + bid * PNUM] - node_temp->Key[i + 1];
							dis = max(dis, dis_lb);
						}
						dis = max(dis, 0.0);
						pushPQ(node_temp->Children[i], dis, -1, pq_c[idx]);
					}
					else if (node_temp->Children[i] == NULL)
					{
						int data_id = node_temp->id[i];
						float result = 0;

						float dis = node_temp->Key[i] - dis_pivot[pivot_flag[idx] + bid * PNUM];
						if (i != node_temp->KeyNum[0] - 1)
						{
							float dis_lb = dis_pivot[pivot_flag[idx] + bid * PNUM] - node_temp->Key[i + 1];
							dis = max(dis, dis_lb);
						}

						if (dis <= dis_k || (!isFullPQK(pq_a[idx])))
						{
							// Compute the real distance
							if (data_info[2] == 2)
							{ // L2 distance
								for (int i = 0; i < data_info[0]; i++)
								{
									result += pow(data_d[data_id * data_info[0] + i] - data_d[qid[bid] * data_info[0] + i], 2);
								}
								result = pow(result, 0.5);
							}
							else if (data_info[2] == 1)
							{ // L1 distance
								for (int i = 0; i < data_info[0]; i++)
								{
									result += abs(data_d[data_id * data_info[0] + i] - data_d[qid[bid] * data_info[0] + i]);
								}
							}
							else if (data_info[2] == 0)
							{ // Max value
								float temp = 0;
								for (int i = 0; i < data_info[0]; i++)
								{
									temp = abs(data_d[data_id * data_info[0] + i] - data_d[qid[bid] * data_info[0] + i]);
									if (temp > result)
										result = temp;
								}
							}
							else if (data_info[2] == 5)
							{
								float sa1 = 0, sa2 = 0, sa3 = 0;
								for (int i = 0; i < data_info[0]; i++)
								{
									sa1 += data_d[data_id * data_info[0] + i] * data_d[data_id * data_info[0] + i];
									sa2 += data_d[qid[bid] * data_info[0] + i] * data_d[qid[bid] * data_info[0] + i];
									sa3 += data_d[data_id * data_info[0] + i] * data_d[qid[bid] * data_info[0] + i];
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
								int m = size_s[qid[bid]];
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
											int cost = (data_s[data_id * MaxC + i - 1] == data_s[qid[bid] * MaxC + j - 1]) ? 0 : 1;
											table[i][j] = 1 + min(table[i - 1][j], table[i][j - 1]);
											table[i][j] = min(table[i - 1][j - 1] + cost, table[i][j]);
										}
									}
									result = table[n][m];
								}
							}

							// Save result
							if (result <= dis_k || (!isFullPQK(pq_a[idx])))
							{
								if (isFullPQK(pq_a[idx]))
								{
									popPQK(pq_a[idx], node_res, dis_res, id_res);
								}
								pushPQK(NULL, result, data_id, pq_a[idx]);
								findMaxPQK(pq_a[idx], node_res, dis_res, id_res);
								dis_k = dis_res;
							}
						}
					}
				}
			}
		}
	}
}

// Get low bounds of knn
// Block -> query
// Thread -> Search one of B+-tree for each query
__global__ void getKnnBound(BPlusNode **T, int qnum, short *data_d, int *data_info, int *qid, int k, PriorityQueue_k **pq_a, PriorityQueue **pq_c,
							float *knn_bound, int *isSatisfied, int *tree_sum_prefix, int *node_sum_prefix, float *dis_pivot, char *data_s, int *size_s)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;

	if (bid < qnum)
	{
		if (tid == 0)
		{
			for (int id = tid; id < PNUM; id++)
			{
				int idx = bid * PNUM + id;
				if (isSatisfied[idx] > 0)
				{
					int tree_idx = tree_sum_prefix[id];	 // Tree idx
					int nid = node_sum_prefix[tree_idx]; // Node idx
					float dis_k = 99999;
					float dis_temp;
					int id_temp;
					BPlusNode *node_temp;
					float dis_res;
					int id_res;
					BPlusNode *node_res;

					initPQK(k, pq_a[bid], 0);
					initPQ(DNUM, pq_c[bid], 1);

					dis_temp = 0;
					id_temp = -1;
					node_temp = T[nid];

					pushPQ(node_temp, dis_temp, id_temp, pq_c[bid]);

					while (!isEmptyPQ(pq_c[bid]))
					{
						popPQ(pq_c[bid], node_temp, dis_temp, id_temp);

						if (dis_temp > dis_k)
							continue;

						for (int i = 0; i < node_temp->KeyNum[0]; i++)
						{
							if (node_temp->Children[i] != NULL)
							{
								float dis = node_temp->Key[i] - dis_pivot[id + bid * PNUM];
								if (i != node_temp->KeyNum[0] - 1)
								{
									float dis_lb = dis_pivot[id + bid * PNUM] - node_temp->Key[i + 1];
									dis = max(dis, dis_lb);
								}
								dis = max(dis, 0.0);
								pushPQ(node_temp->Children[i], dis, -1, pq_c[bid]);
							}
							else if (node_temp->Children[i] == NULL)
							{
								int data_id = node_temp->id[i];
								float result = 0;

								float dis = node_temp->Key[i] - dis_pivot[id + bid * PNUM];
								if (i != node_temp->KeyNum[0] - 1)
								{
									float dis_lb = dis_pivot[id + bid * PNUM] - node_temp->Key[i + 1];
									dis = max(dis, dis_lb);
								}

								if (dis <= dis_k || (!isFullPQK(pq_a[bid])))
								{
									// Compute the real distance
									if (data_info[2] == 2)
									{ // L2 distance
										for (int i = 0; i < data_info[0]; i++)
										{
											result += pow(data_d[data_id * data_info[0] + i] - data_d[qid[bid] * data_info[0] + i], 2);
										}
										result = pow(result, 0.5);
									}
									else if (data_info[2] == 1)
									{ // L1 distance
										for (int i = 0; i < data_info[0]; i++)
										{
											result += abs(data_d[data_id * data_info[0] + i] - data_d[qid[bid] * data_info[0] + i]);
										}
									}
									else if (data_info[2] == 0)
									{ // Max value
										float temp = 0;
										for (int i = 0; i < data_info[0]; i++)
										{
											temp = abs(data_d[data_id * data_info[0] + i] - data_d[qid[bid] * data_info[0] + i]);
											if (temp > result)
												result = temp;
										}
									}
									else if (data_info[2] == 5)
									{
										float sa1 = 0, sa2 = 0, sa3 = 0;
										for (int i = 0; i < data_info[0]; i++)
										{
											sa1 += data_d[data_id * data_info[0] + i] * data_d[data_id * data_info[0] + i];
											sa2 += data_d[qid[bid] * data_info[0] + i] * data_d[qid[bid] * data_info[0] + i];
											sa3 += data_d[data_id * data_info[0] + i] * data_d[qid[bid] * data_info[0] + i];
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
										int m = size_s[qid[bid]];
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
													int cost = (data_s[data_id * MaxC + i - 1] == data_s[qid[bid] * MaxC + j - 1]) ? 0 : 1;
													table[i][j] = 1 + min(table[i - 1][j], table[i][j - 1]);
													table[i][j] = min(table[i - 1][j - 1] + cost, table[i][j]);
												}
											}
											result = table[n][m];
										}
									}

									// Save result
									if (result <= dis_k || (!isFullPQK(pq_a[bid])))
									{
										if (isFullPQK(pq_a[bid]))
										{
											popPQK(pq_a[bid], node_res, dis_res, id_res);
										}
										pushPQK(NULL, result, data_id, pq_a[bid]);
										findMaxPQK(pq_a[bid], node_res, dis_res, id_res);
										dis_k = dis_res;
									}
								}
							}
						}
					}

					findMaxPQK(pq_a[bid], node_res, dis_res, id_res);
					knn_bound[bid] = dis_res;
				}
				break;
			}
		}
	}
}

// Compute info about search root idx
__global__ void getCounter(int *qnum_counter, int *isSatisfied, int *tree_counter, int qnum)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int total_num = gridDim.x * blockDim.x;

	for (int idx = id; idx < qnum; idx += total_num)
	{
		int end = (idx + 1) * PNUM - 1;
		int result = isSatisfied[end] + tree_counter[end];

		qnum_counter[idx] = result;
	}
}

// Merge root node
// Block -> query
// Thread -> partition of each query
__global__ void mergeRoot(int *isSatisfied, int *tree_counter, int qnum, int *qnum_counter_prefix, int *root_idx,
						  int *tree_sum_prefix, int *node_sum_prefix, int *pivot_flag)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;

	if (bid < qnum)
	{
		for (int id = tid; id < PNUM; id += blockDim.x)
		{
			int idx = tree_counter[bid * PNUM + id] + qnum_counter_prefix[bid];
			int tree_idx = tree_sum_prefix[id];	 // Tree idx
			int nid = node_sum_prefix[tree_idx]; // Node idx

			for (int i = 0; i < isSatisfied[bid * PNUM + id]; i++)
			{
				root_idx[idx] = nid;
				pivot_flag[idx] = id;
				idx++;
				tree_idx++;
				nid = node_sum_prefix[tree_idx];
			}
		}
	}
}

// Get distance bounds for knn
// Block -> query
// Thread -> pivot of each partition, compute the distance between pivot and query
__global__ void getDisBound(float *dis_pivot, int qnum, float *radius, float *dis_lb, float *dis_ub)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;

	if (bid < qnum)
	{
		for (int id = tid; id < PNUM; id += blockDim.x)
		{
			int idx = bid * PNUM + id;
			float dis_l = dis_pivot[idx] - radius[id];
			float dis_u = dis_pivot[idx] + radius[id];
			dis_lb[idx] = dis_l;
			dis_ub[idx] = dis_u;
		}
	}
}

// Merge the result of knn
// Block -> query
// Thread -> B+-tree for each query
__global__ void mergeKnn(int k, PriorityQueue_k **pq_a, float *dis_knn, int *id_knn, int qnum, int *qnum_counter,
						 int *qnum_counter_prefix)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;

	if (bid < qnum)
	{
		for (int id = tid; id < qnum_counter[bid]; id += blockDim.x)
		{
			int idx = qnum_counter_prefix[bid] + id;
			float dis_temp;
			int id_temp;
			BPlusNode *node_temp;

			for (int j = 0; j < k; j++)
			{
				if (isEmptyPQK(pq_a[idx]))
				{
					dis_knn[idx * k + j] = 99999;
					id_knn[idx * k + j] = -1;
				}
				else
				{
					popPQK(pq_a[idx], node_temp, dis_temp, id_temp);
					dis_knn[idx * k + j] = dis_temp;
					id_knn[idx * k + j] = id_temp;
				}
			}
		}
		/*__syncthreads();

		if (tid == 0) {
			int start_idx = qnum_counter_prefix[bid] * k;
			int end_idx = (qnum_counter_prefix[bid] + qnum_counter[bid]) * k;
			thrust::sort_by_key(thrust::device, dis_knn + start_idx, dis_knn + end_idx, id_knn + start_idx);
		}*/
	}
}

// Show results for knn
__global__ void showSearchKnn(int qnum, int *qnum_counter, int *qnum_counter_prefix, int k, float *dis_knn, int *id_knn)
{
	for (int m = qnum - 5; m < qnum; m++)
	{
		int id = qnum_counter_prefix[m];
		for (int j = 0; j < k; j++)
		{
			printf("%d, %f;  ", id_knn[id * k + j], dis_knn[id * k + j]);
		}
		printf("\n");
	}
}

// Show results for rnn
__global__ void showSearchRnn(int qnum, Obj init_result, int *result_counter, int *qnum_counter, int *qnum_counter_prefix)
{
	// float size = 0;

	for (int k = qnum - 5; k < qnum; k++)
	{
		for (int i = 0; i < qnum_counter[k]; i++)
		{
			int id = qnum_counter_prefix[k] + i;
			// printf("%d\n", result_counter[id]);
			// size += result_counter[id];
			for (int j = 0; j < result_counter[id]; j++)
			{
				int idx = j + (id)*DNUM;
				printf("%f, %d;  ", init_result.dis[idx], init_result.res_id[idx]);
			}
		}
		printf("\n");
	}

	// float avg_size = size / qnum;
	// printf("Average size of search result: %.2f\n", avg_size);
}

// ���еļ���isSatisfied
__global__ void getSatisfied(int *isSatisfied, int *tree_counter, int qnum)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int total_num = gridDim.x * blockDim.x;

	for (int idx = id; idx < qnum; idx += total_num)
	{
		int start = idx * PNUM;
		int end = (idx + 1) * PNUM;
		thrust::exclusive_scan(thrust::device, isSatisfied + start, isSatisfied + end, tree_counter + start); // in-place scan
	}
}

void search(short *data_d, int *pid, int *qid, int qnum, float *dis_pivot, int *data_info, float r, float *radius,
			int *&isSatisfied, int *&tree_num, BPlusNode **T, int *&tree_sum_prefix, int *&node_sum_prefix,
			int *&qnum_counter, int *&qnum_counter_prefix, int *&result_counter, Obj &init_result, int search_type, int k,
			float *&dis_knn, int *&id_knn, char *data_s, int *size_s)
{
	cout << "Search..." << endl;

	int *tree_counter;
	int *root_idx;
	int *root_idx_f;
	int *tree_filter;
	int *tree_filter_prefix;
	SeqQueue **SQ;
	float *dis_lb;
	float *dis_ub;
	int *pivot_flag;
	int *pivot_flag_f;
	PriorityQueue **pq_c;
	PriorityQueue_k **pq_a;
	PriorityQueue **pq_cb;
	PriorityQueue_k **pq_ab;
	float *knn_bound;

	cudaMalloc((void **)&tree_counter, PNUM * qnum * sizeof(int));
	cudaMallocManaged((void **)&qnum_counter, qnum * sizeof(int));
	cudaMallocManaged((void **)&qnum_counter_prefix, qnum * sizeof(int));

	getPivotDis<<<qnum, THREAD_NUM>>>(data_d, pid, qid, qnum, dis_pivot, data_info, data_s, size_s);
	cudaDeviceSynchronize();

	if (search_type == 0)
	{
		cudaMalloc((void **)&dis_lb, PNUM * qnum * sizeof(float));
		cudaMalloc((void **)&dis_ub, PNUM * qnum * sizeof(float));
		cudaMalloc((void **)&knn_bound, qnum * sizeof(float));
		CHECK(cudaMallocManaged((void **)&pq_cb, qnum * sizeof(PriorityQueue *)));
		CHECK(cudaMallocManaged((void **)&pq_ab, qnum * sizeof(PriorityQueue_k *)));
		for (int i = 0; i < qnum; i++)
		{
			CHECK(cudaMalloc((void **)&pq_ab[i], sizeof(PriorityQueue_k)));
			CHECK(cudaMalloc((void **)&pq_cb[i], sizeof(PriorityQueue)));
		}

		getDisBound<<<qnum, THREAD_NUM>>>(dis_pivot, qnum, radius, dis_lb, dis_ub);
		cudaDeviceSynchronize();
		cudaError_t cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
			fprintf(stderr, "getDisBound error: %s\n", cudaGetErrorString(cudaStatus));

		for (int i = 0; i < qnum; i++)
		{
			int start = i * PNUM;
			int end = (i + 1) * PNUM;
			thrust::sort(thrust::device, dis_ub + start, dis_ub + end);
		}

		pivotFilterKnn<<<qnum, THREAD_NUM>>>(qnum, dis_ub, dis_lb, isSatisfied, tree_num, k);
		cudaDeviceSynchronize();
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
			fprintf(stderr, "pivotFilterKnn error: %s\n", cudaGetErrorString(cudaStatus));

		getKnnBound<<<qnum, 1>>>(T, qnum, data_d, data_info, qid, k, pq_ab, pq_cb, knn_bound, isSatisfied, tree_sum_prefix, node_sum_prefix,
								 dis_pivot, data_s, size_s);
		cudaDeviceSynchronize();
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
			fprintf(stderr, "getKnnBound error: %s\n", cudaGetErrorString(cudaStatus));

		for (int i = 0; i < qnum; i++)
		{
			cudaFree(pq_ab[i]);
		}
		cudaFree(pq_ab);
		for (int i = 0; i < qnum; i++)
		{
			cudaFree(pq_cb[i]);
		}
		cudaFree(pq_cb);

		pivotFilterKnnAgain<<<qnum, THREAD_NUM>>>(qnum, knn_bound, dis_lb, isSatisfied, tree_num);
		cudaDeviceSynchronize();
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
			fprintf(stderr, "pivotFilterKnnAgain error: %s\n", cudaGetErrorString(cudaStatus));

		// for (int i = 0; i < qnum; i++) {
		//	int start = i * PNUM;
		//	int end = (i + 1) * PNUM;
		//	thrust::exclusive_scan(thrust::device, isSatisfied + start, isSatisfied + end, tree_counter + start); // in-place scan
		// }
		getSatisfied<<<(qnum - 1) / THREAD_NUM + 1, THREAD_NUM>>>(isSatisfied, tree_counter, qnum);
		cudaDeviceSynchronize();

		int tree_num_block = (qnum - 1) / THREAD_NUM + 1;
		getCounter<<<tree_num_block, THREAD_NUM>>>(qnum_counter, isSatisfied, tree_counter, qnum);
		cudaDeviceSynchronize();
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
			fprintf(stderr, "getCounter error: %s\n", cudaGetErrorString(cudaStatus));

		thrust::exclusive_scan(thrust::device, qnum_counter, qnum_counter + qnum, qnum_counter_prefix);
		int total_c = thrust::reduce(thrust::device, qnum_counter, qnum_counter + qnum, 0);
		cout << "Total search tree num: " << total_c << endl;

		cudaMalloc((void **)&pivot_flag, total_c * sizeof(int));
		cudaMalloc((void **)&root_idx, total_c * sizeof(int));
		cudaMalloc((void **)&tree_filter, total_c * sizeof(int));
		cudaMalloc((void **)&tree_filter_prefix, total_c * sizeof(int));

		mergeRoot<<<qnum, THREAD_NUM>>>(isSatisfied, tree_counter, qnum, qnum_counter_prefix, root_idx,
										tree_sum_prefix, node_sum_prefix, pivot_flag);
		cudaDeviceSynchronize();
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
			fprintf(stderr, "mergeRoot error: %s\n", cudaGetErrorString(cudaStatus));

		treeFilterKnn<<<qnum, THREAD_NUM>>>(root_idx, T, qnum_counter, qnum_counter_prefix, qnum, dis_pivot, pivot_flag,
											qid, knn_bound, tree_filter);
		cudaDeviceSynchronize();
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
			fprintf(stderr, "treeFilterKnn error: %s\n", cudaGetErrorString(cudaStatus));

		thrust::exclusive_scan(thrust::device, qnum_counter, qnum_counter + qnum, qnum_counter_prefix);
		thrust::exclusive_scan(thrust::device, tree_filter, tree_filter + total_c, tree_filter_prefix); // in-place scan
		int total_c_f = thrust::reduce(thrust::device, qnum_counter, qnum_counter + qnum, 0);
		cout << "Total search tree num: " << total_c_f << endl;

		cudaMalloc((void **)&pivot_flag_f, total_c_f * sizeof(int));
		cudaMalloc((void **)&root_idx_f, total_c_f * sizeof(int));

		int bn = (total_c - 1) / THREAD_NUM + 1;
		mergeTree<<<bn, THREAD_NUM>>>(tree_filter_prefix, tree_filter, total_c, root_idx, root_idx_f, pivot_flag, pivot_flag_f);
		cudaDeviceSynchronize();
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
			fprintf(stderr, "mergeTree error: %s\n", cudaGetErrorString(cudaStatus));

		cudaFree(root_idx);
		cudaFree(pivot_flag);
		cudaFree(isSatisfied);
		cudaFree(tree_counter);
		cudaFree(tree_num);
		cudaFree(tree_sum_prefix);
		cudaFree(node_sum_prefix);
		cudaFree(dis_ub);
		cudaFree(dis_lb);

		CHECK(cudaMallocManaged((void **)&pq_c, total_c_f * sizeof(PriorityQueue *)));
		CHECK(cudaMallocManaged((void **)&pq_a, total_c_f * sizeof(PriorityQueue_k *)));
		for (int i = 0; i < total_c_f; i++)
		{
			CHECK(cudaMalloc((void **)&pq_a[i], sizeof(PriorityQueue_k)));
			CHECK(cudaMalloc((void **)&pq_c[i], sizeof(PriorityQueue)));
		}

		searchKnnD<<<qnum, THREAD_NUM>>>(root_idx_f, T, qnum_counter, qnum_counter_prefix, qnum, dis_pivot, pivot_flag_f,
										 data_d, data_info, qid, k, pq_a, pq_c, knn_bound, data_s, size_s);
		cudaDeviceSynchronize();
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
			fprintf(stderr, "searchKnnD error: %s\n", cudaGetErrorString(cudaStatus));

		cudaFree(knn_bound);
		for (int i = 0; i < total_c_f; i++)
		{
			cudaFree(pq_c[i]);
		}
		cudaFree(pq_c);

		CHECK(cudaMalloc((void **)&dis_knn, total_c_f * k * sizeof(float)));
		CHECK(cudaMalloc((void **)&id_knn, total_c_f * k * sizeof(int)));

		mergeKnn<<<qnum, THREAD_NUM>>>(k, pq_a, dis_knn, id_knn, qnum, qnum_counter, qnum_counter_prefix);
		cudaDeviceSynchronize();
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
			fprintf(stderr, "mergeKnn error: %s\n", cudaGetErrorString(cudaStatus));
		for (int i = 0; i < qnum; i++)
		{
			int start_idx = qnum_counter_prefix[i] * k;
			int end_idx = start_idx + qnum_counter[i] * k;
			thrust::sort_by_key(thrust::device, dis_knn + start_idx, dis_knn + end_idx, id_knn + start_idx);
		}

		cudaFree(root_idx_f);
		cudaFree(pivot_flag_f);
		for (int i = 0; i < total_c_f; i++)
		{
			cudaFree(pq_a[i]);
		}
		cudaFree(pq_a);
	}
	else
	{
		pivotFilterRnn<<<qnum, THREAD_NUM>>>(dis_pivot, qnum, r, radius, isSatisfied, tree_num);
		cudaDeviceSynchronize();

		// for (int i = 0; i < qnum; i++) {
		//	int start = i * PNUM;
		//	int end = (i + 1) * PNUM;
		//	thrust::exclusive_scan(thrust::device, isSatisfied + start, isSatisfied + end, tree_counter + start); // in-place scan
		// }
		getSatisfied<<<(qnum - 1) / THREAD_NUM + 1, THREAD_NUM>>>(isSatisfied, tree_counter, qnum);
		cudaDeviceSynchronize();

		int tree_num_block = (qnum - 1) / THREAD_NUM + 1;
		getCounter<<<tree_num_block, THREAD_NUM>>>(qnum_counter, isSatisfied, tree_counter, qnum);
		cudaDeviceSynchronize();

		thrust::exclusive_scan(thrust::device, qnum_counter, qnum_counter + qnum, qnum_counter_prefix);
		int total_c = thrust::reduce(thrust::device, qnum_counter, qnum_counter + qnum, 0);
		cout << "Total search tree num: " << total_c << endl;
		cudaMalloc((void **)&pivot_flag, total_c * sizeof(int));
		cudaMalloc((void **)&root_idx, total_c * sizeof(int));
		cudaMalloc((void **)&tree_filter, total_c * sizeof(int));
		cudaMalloc((void **)&tree_filter_prefix, total_c * sizeof(int));

		mergeRoot<<<qnum, THREAD_NUM>>>(isSatisfied, tree_counter, qnum, qnum_counter_prefix, root_idx,
										tree_sum_prefix, node_sum_prefix, pivot_flag);
		cudaDeviceSynchronize();

		treeFilterRnn<<<qnum, THREAD_NUM>>>(root_idx, T, qnum_counter, qnum_counter_prefix, qnum, dis_pivot, pivot_flag,
											qid, r, tree_filter);
		cudaDeviceSynchronize();

		thrust::exclusive_scan(thrust::device, qnum_counter, qnum_counter + qnum, qnum_counter_prefix);
		thrust::exclusive_scan(thrust::device, tree_filter, tree_filter + total_c, tree_filter_prefix); // in-place scan
		int total_c_f = thrust::reduce(thrust::device, qnum_counter, qnum_counter + qnum, 0);
		cout << "Total search tree num: " << total_c_f << endl;

		cudaMalloc((void **)&pivot_flag_f, total_c_f * sizeof(int));
		cudaMalloc((void **)&root_idx_f, total_c_f * sizeof(int));

		int bn = (total_c - 1) / THREAD_NUM + 1;
		mergeTree<<<bn, THREAD_NUM>>>(tree_filter_prefix, tree_filter, total_c, root_idx, root_idx_f, pivot_flag, pivot_flag_f);
		cudaDeviceSynchronize();

		cudaFree(root_idx);
		cudaFree(pivot_flag);
		cudaFree(isSatisfied);
		cudaFree(tree_counter);
		cudaFree(tree_num);
		cudaFree(tree_sum_prefix);
		cudaFree(node_sum_prefix);

		CHECK(cudaMalloc((void **)&result_counter, total_c_f * sizeof(int)));
		CHECK(cudaMalloc((void **)&init_result.res_id, DNUM * total_c_f * sizeof(int)));
		CHECK(cudaMalloc((void **)&init_result.dis, DNUM * total_c_f * sizeof(float)));
		CHECK(cudaMallocManaged((void **)&SQ, total_c_f * sizeof(SeqQueue *)));
		for (int i = 0; i < total_c_f; i++)
		{
			CHECK(cudaMalloc((void **)&SQ[i], sizeof(SeqQueue)));
		}

		searchRnnD<<<qnum, THREAD_NUM>>>(root_idx_f, T, qnum_counter, qnum_counter_prefix, qnum,
										 SQ, dis_pivot, pivot_flag_f, r, data_d, data_info, qid, init_result, result_counter, data_s, size_s);
		cudaDeviceSynchronize();
		cudaError_t cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "searchRnnD error: %s\n", cudaGetErrorString(cudaStatus));
			// goto Error;
		}

		for (int i = 0; i < total_c_f; i++)
		{
			cudaFree(SQ[i]);
		}
		cudaFree(SQ);
		cudaFree(root_idx_f);
		cudaFree(pivot_flag_f);
	}
}