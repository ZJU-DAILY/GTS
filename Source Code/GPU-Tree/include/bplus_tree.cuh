// GPU B+-tree index
// Created on 24-01-05

#pragma once
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <thrust/sort.h>
#include "partition.cuh"
#include "config.cuh"

#define M (4) // order of B+-tree
#define LIMIT_M_2 (M % 2 ? (M + 1) / 2 : M / 2)
#define Unavailable INT_MIN
#define DNUM 500 // data num of B+-tree

// typedef struct BPlusNode* BPlusTree, * Position;
typedef float KeyType;
typedef struct BPlusNode
{
	int idx[1];
	int KeyNum[1];
	int id[M + 1];
	KeyType Key[M + 1];
	BPlusNode *Children[M + 1];
	BPlusNode *Next;
};

__device__ BPlusNode *FindMostLeft(BPlusNode *P)
{
	BPlusNode *Tmp;

	Tmp = P;

	while (Tmp != NULL && Tmp->Children[0] != NULL)
	{
		Tmp = Tmp->Children[0];
	}
	return Tmp;
}

__device__ BPlusNode *FindMostRight(BPlusNode *P)
{
	BPlusNode *Tmp;

	Tmp = P;

	while (Tmp != NULL && Tmp->Children[Tmp->KeyNum[0] - 1] != NULL)
	{
		Tmp = Tmp->Children[Tmp->KeyNum[0] - 1];
	}
	return Tmp;
}

__device__ BPlusNode *FindSibling(BPlusNode *Parent, int i)
{
	BPlusNode *Sibling;
	int Limit;

	Limit = M;

	Sibling = NULL;
	if (i == 0)
	{
		if (Parent->Children[1]->KeyNum[0] < Limit)
			Sibling = Parent->Children[1];
	}
	else if (Parent->Children[i - 1]->KeyNum[0] < Limit)
		Sibling = Parent->Children[i - 1];
	else if (i + 1 < Parent->KeyNum[0] && Parent->Children[i + 1]->KeyNum[0] < Limit)
	{
		Sibling = Parent->Children[i + 1];
	}

	return Sibling;
}

__device__ BPlusNode *FindSiblingKeyNum_M_2(BPlusNode *Parent, int i, int *j)
{
	int Limit;
	BPlusNode *Sibling;
	Sibling = NULL;

	Limit = LIMIT_M_2;

	if (i == 0)
	{
		if (Parent->Children[1]->KeyNum[0] > Limit)
		{
			Sibling = Parent->Children[1];
			*j = 1;
		}
	}
	else
	{
		if (Parent->Children[i - 1]->KeyNum[0] > Limit)
		{
			Sibling = Parent->Children[i - 1];
			*j = i - 1;
		}
		else if (i + 1 < Parent->KeyNum[0] && Parent->Children[i + 1]->KeyNum[0] > Limit)
		{
			Sibling = Parent->Children[i + 1];
			*j = i + 1;
		}
	}

	return Sibling;
}

__device__ BPlusNode *InsertElement(int isKey, BPlusNode *Parent, BPlusNode *X, KeyType Key, int id, int i, int j)
{

	int k;
	if (isKey)
	{
		k = X->KeyNum[0] - 1;
		while (k >= j)
		{
			X->Key[k + 1] = X->Key[k];
			X->id[k + 1] = X->id[k];
			k--;
		}

		X->Key[j] = Key;
		X->id[j] = id;

		if (Parent != NULL)
		{
			Parent->Key[i] = X->Key[0];
			Parent->id[i] = X->id[0];
		}

		X->KeyNum[0]++;
	}
	else
	{
		if (X->Children[0] == NULL)
		{
			if (i > 0)
				Parent->Children[i - 1]->Next = X;
			X->Next = Parent->Children[i];
		}

		k = Parent->KeyNum[0] - 1;
		while (k >= i)
		{
			Parent->Children[k + 1] = Parent->Children[k];
			Parent->Key[k + 1] = Parent->Key[k];
			Parent->id[k + 1] = Parent->id[k];
			k--;
		}
		Parent->Key[i] = X->Key[0];
		Parent->id[i] = X->id[0];
		Parent->Children[i] = X;

		Parent->KeyNum[0]++;
	}
	return X;
}

__device__ BPlusNode *RemoveElement(int isKey, BPlusNode *Parent, BPlusNode *X, int i, int j)
{

	int k, Limit;

	if (isKey)
	{
		Limit = X->KeyNum[0];
		k = j + 1;
		while (k < Limit)
		{
			X->Key[k - 1] = X->Key[k];
			X->id[k - 1] = X->id[k];
			k++;
		}

		X->Key[X->KeyNum[0] - 1] = Unavailable;
		X->id[X->KeyNum[0] - 1] = Unavailable;

		Parent->Key[i] = X->Key[0];
		Parent->id[i] = X->id[0];

		X->KeyNum[0]--;
	}
	else
	{
		if (X->Children[0] == NULL && i > 0)
		{
			Parent->Children[i - 1]->Next = Parent->Children[i + 1];
		}
		Limit = Parent->KeyNum[0];
		k = i + 1;
		while (k < Limit)
		{
			Parent->Children[k - 1] = Parent->Children[k];
			Parent->Key[k - 1] = Parent->Key[k];
			Parent->id[k - 1] = Parent->id[k];
			k++;
		}

		// Parent->Key[i] = Parent->Children[i]->Key[0];
		Parent->Children[Parent->KeyNum[0] - 1] = NULL;
		Parent->Key[Parent->KeyNum[0] - 1] = Unavailable;
		Parent->id[Parent->KeyNum[0] - 1] = Unavailable;

		Parent->KeyNum[0]--;
	}
	return X;
}

__device__ BPlusNode *MoveElement(BPlusNode *Src, BPlusNode *Dst, BPlusNode *Parent, int i, int n)
{
	KeyType TmpKey;
	int TmpId;
	BPlusNode *Child;
	int j, SrcInFront;

	SrcInFront = 0;

	if (Src->Key[0] < Dst->Key[0])
		SrcInFront = 1;

	j = 0;
	if (SrcInFront)
	{
		if (Src->Children[0] != NULL)
		{
			while (j < n)
			{
				Child = Src->Children[Src->KeyNum[0] - 1];
				RemoveElement(0, Src, Child, Src->KeyNum[0] - 1, Unavailable);
				InsertElement(0, Dst, Child, Unavailable, Unavailable, 0, Unavailable);
				j++;
			}
		}
		else
		{
			while (j < n)
			{
				TmpKey = Src->Key[Src->KeyNum[0] - 1];
				TmpId = Src->id[Src->KeyNum[0] - 1];
				RemoveElement(1, Parent, Src, i, Src->KeyNum[0] - 1);
				InsertElement(1, Parent, Dst, TmpKey, TmpId, i + 1, 0);
				j++;
			}
		}

		Parent->Key[i + 1] = Dst->Key[0];
		Parent->id[i + 1] = Dst->id[0];

		if (Src->KeyNum[0] > 0)
			FindMostRight(Src)->Next = FindMostLeft(Dst);
	}
	else
	{
		if (Src->Children[0] != NULL)
		{
			while (j < n)
			{
				Child = Src->Children[0];
				RemoveElement(0, Src, Child, 0, Unavailable);
				InsertElement(0, Dst, Child, Unavailable, Unavailable, Dst->KeyNum[0], Unavailable);
				j++;
			}
		}
		else
		{
			while (j < n)
			{
				TmpKey = Src->Key[0];
				TmpId = Src->id[0];
				RemoveElement(1, Parent, Src, i, 0);
				InsertElement(1, Parent, Dst, TmpKey, TmpId, i - 1, Dst->KeyNum[0]);
				j++;
			}
		}

		Parent->Key[i] = Src->Key[0];
		Parent->id[i] = Src->id[0];
		if (Src->KeyNum[0] > 0)
			FindMostRight(Dst)->Next = FindMostLeft(Src);
	}

	return Parent;
}

__device__ BPlusNode *SplitNode(BPlusNode *Parent, BPlusNode *X, int i, int *node_idx, int tree_idx, BPlusNode **T)
{
	int j, k, Limit;

	node_idx[tree_idx] = node_idx[tree_idx] + 1;
	BPlusNode *NewNode = T[node_idx[tree_idx]];

	// NewNode = mallocNewNode();
	// mallocNewNode(NewNode);

	k = 0;
	j = X->KeyNum[0] / 2;
	Limit = X->KeyNum[0];
	while (j < Limit)
	{
		if (X->Children[0] != NULL)
		{
			NewNode->Children[k] = X->Children[j];
			X->Children[j] = NULL;
		}
		NewNode->Key[k] = X->Key[j];
		NewNode->id[k] = X->id[j];
		X->Key[j] = Unavailable;
		X->id[j] = Unavailable;
		NewNode->KeyNum[0]++;
		X->KeyNum[0]--;
		j++;
		k++;
	}

	if (Parent != NULL)
		InsertElement(0, Parent, NewNode, Unavailable, Unavailable, i + 1, Unavailable);
	else
	{
		node_idx[tree_idx] = node_idx[tree_idx] + 1;
		Parent = T[node_idx[tree_idx]];
		// Parent = mallocNewNode();
		// mallocNewNode(Parent);
		InsertElement(0, Parent, X, Unavailable, Unavailable, 0, Unavailable);
		InsertElement(0, Parent, NewNode, Unavailable, Unavailable, 1, Unavailable);

		return Parent;
	}

	return X;
}

__device__ BPlusNode *MergeNode(BPlusNode *Parent, BPlusNode *X, BPlusNode *S, int i, bool &merge_flag, int &merge_idx)
{
	int Limit;

	if (S->KeyNum[0] > LIMIT_M_2)
	{
		MoveElement(S, X, Parent, i, 1);
	}
	else
	{
		Limit = X->KeyNum[0];
		MoveElement(X, S, Parent, i, Limit);
		RemoveElement(0, Parent, X, i, Unavailable);

		int m = 0;
		while (m < M + 1)
		{
			X->Key[m] = Unavailable;
			X->id[m] = Unavailable;
			X->Children[m] = NULL;
			m++;
		}
		X->Next = NULL;
		X->KeyNum[0] = 0;

		merge_flag = true;
		merge_idx = X->idx[0];

		// free(X);
		// X = NULL;
	}

	return Parent;
}

__device__ BPlusNode *RecursiveInsert(BPlusNode *T, KeyType Key, int id, int i, BPlusNode *Parent, int *node_idx, int tree_idx, BPlusNode **TN)
{
	int j, Limit;
	BPlusNode *Sibling;

	j = 0;
	while (j < T->KeyNum[0] && Key >= T->Key[j])
	{
		j++;
	}
	if (j != 0 && T->Children[0] != NULL)
		j--;

	if (T->Children[0] == NULL)
		T = InsertElement(1, Parent, T, Key, id, i, j);
	else
		T->Children[j] = RecursiveInsert(T->Children[j], Key, id, j, T, node_idx, tree_idx, TN);

	Limit = M;

	if (T->KeyNum[0] > Limit)
	{
		if (Parent == NULL)
		{
			T = SplitNode(Parent, T, i, node_idx, tree_idx, TN);
		}
		else
		{
			Sibling = FindSibling(Parent, i);
			if (Sibling != NULL)
			{
				MoveElement(T, Sibling, Parent, i, 1);
			}
			else
			{
				T = SplitNode(Parent, T, i, node_idx, tree_idx, TN);
			}
		}
	}

	if (Parent != NULL)
	{
		Parent->Key[i] = T->Key[0];
		Parent->id[i] = T->id[0];
	}

	return T;
}

__device__ BPlusNode *Insert(BPlusNode *T, KeyType Key, int id, int *node_idx, int tree_idx, BPlusNode **TN)
{
	return RecursiveInsert(T, Key, id, 0, NULL, node_idx, tree_idx, TN);
}

__device__ BPlusNode *RecursiveRemove(BPlusNode *T, KeyType Key, int i, BPlusNode *Parent, bool &remove_flag, int &remove_idx, bool &merge_flag,
									  int &merge_idx)
{

	int j, NeedAdjust;
	BPlusNode *Sibling, *Tmp;

	Sibling = NULL;

	j = 0;
	while (j < T->KeyNum[0] && Key >= T->Key[j])
	{
		if (Key == T->Key[j])
			break;
		j++;
	}

	if (T->Children[0] == NULL)
	{
		if (Key != T->Key[j] || j == T->KeyNum[0])
			return T;
	}
	else if (j == T->KeyNum[0] || Key < T->Key[j])
		j--;

	if (T->Children[0] == NULL)
	{
		T = RemoveElement(1, Parent, T, i, j);
	}
	else
	{
		T->Children[j] = RecursiveRemove(T->Children[j], Key, j, T, remove_flag, remove_idx, merge_flag, merge_idx);
	}

	NeedAdjust = 0;
	if (Parent == NULL && T->Children[0] != NULL && T->KeyNum[0] < 2)
		NeedAdjust = 1;
	else if (Parent != NULL && T->Children[0] != NULL && T->KeyNum[0] < LIMIT_M_2)
		NeedAdjust = 1;
	else if (Parent != NULL && T->Children[0] == NULL && T->KeyNum[0] < LIMIT_M_2)
		NeedAdjust = 1;

	if (NeedAdjust)
	{
		if (Parent == NULL)
		{
			if (T->Children[0] != NULL && T->KeyNum[0] < 2)
			{
				Tmp = T;
				T = T->Children[0];

				int m = 0;
				while (m < M + 1)
				{
					Tmp->Key[m] = Unavailable;
					Tmp->Children[m] = NULL;
					m++;
				}
				Tmp->Next = NULL;
				Tmp->KeyNum[0] = 0;

				remove_flag = true;
				remove_idx = Tmp->idx[0];

				// free(Tmp);
				// Tmp = NULL;

				return T;
			}
		}
		else
		{
			Sibling = FindSiblingKeyNum_M_2(Parent, i, &j);
			if (Sibling != NULL)
			{
				MoveElement(Sibling, T, Parent, j, 1);
			}
			else
			{
				if (i == 0)
					Sibling = Parent->Children[1];
				else
					Sibling = Parent->Children[i - 1];

				Parent = MergeNode(Parent, T, Sibling, i, merge_flag, merge_idx);
				T = Parent->Children[i];
			}
		}
	}

	if (Parent != NULL)
		Parent->Key[i] = T->Key[0];

	return T;
}

__device__ BPlusNode *Remove(BPlusNode *T, KeyType Key, bool &remove_flag, int &remove_idx, bool &merge_flag, int &merge_idx)
{
	return RecursiveRemove(T, Key, 0, NULL, remove_flag, remove_idx, merge_flag, merge_idx);
}

__device__ void Destroy(BPlusNode *&T)
{
	int i, j;
	if (T != NULL)
	{
		i = 0;
		while (i < T->KeyNum[0] + 1)
		{
			Destroy(T->Children[i]);
			i++;
		}

		free(T);
		T = NULL;
	}
}

__device__ void RecursiveTravel(BPlusNode *T, int Level)
{
	int i;
	if (T != NULL)
	{
		printf("  ");
		printf("[Level:%d]-->", Level);
		printf("%d, ", T->idx[0]);
		printf("(");
		i = 0;
		while (i < T->KeyNum[0])
		{ /*  T->Key[i] != Unavailable*/
			printf("%.2f, ", T->Key[i]);
			printf("%d: ", T->id[i]);

			i++;
		}
		printf(")");

		Level++;

		i = 0;
		while (i <= T->KeyNum[0])
		{
			RecursiveTravel(T->Children[i], Level);
			i++;
		}
	}
}

__device__ void Travel(BPlusNode *T)
{
	// printf("hhh\n");
	RecursiveTravel(T, 0);
	printf("\n");
}

__device__ void TravelData(BPlusNode *T)
{
	BPlusNode *Tmp;
	int i;
	if (T == NULL)
		return;
	printf("All Data:");
	Tmp = T;
	while (Tmp->Children[0] != NULL)
		Tmp = Tmp->Children[0];
	while (Tmp != NULL)
	{
		i = 0;
		while (i < Tmp->KeyNum[0])
		{
			printf(" %d,", Tmp->Key[i]);
			printf(" %d: ", Tmp->id[i]);
			i++;
		}
		Tmp = Tmp->Next;
	}
}

__global__ void travelTree(BPlusNode **T, int idx)
{
	Travel(T[idx]);

	for (int i = 0; i < T[9]->KeyNum[0]; i++)
	{
		printf("%.2f, ", T[9]->Key[i]);
		printf("%d: ", T[9]->id[i]);
	}
}

// Construction of B+-tree
__global__ void getTree(BPlusNode **T, int *part_num, int *tree_sum_prefix, Obj obj_m, int pnum,
						int *tree_num, int *node_sum_prefix, int *node_idx, int *part_num_prefix)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;

	if (bid < pnum)
	{

		for (int id = tid; id < tree_num[bid]; id += blockDim.x)
		{
			int num;										// Data num to process for thread
			int data_idx = DNUM * id;						// Data idx
			int tree_idx = tree_sum_prefix[bid] + id;		// Tree idx
			node_idx[tree_idx] = node_sum_prefix[tree_idx]; // Node idx

			if (id == tree_num[bid] - 1)
			{
				num = part_num[bid] - DNUM * (tree_num[bid] - 1);
			}
			else
			{
				num = DNUM;
			}

			int nid = node_idx[tree_idx];
			/*bool remove_flag = false;
			bool merge_flag = false;
			int remove_idx;
			int merge_idx;*/

			for (int i = 0; i < num; i++)
			{
				// if (obj_m[bid].res_id[data_idx + i] == 77) printf("id = 77\n");
				T[nid] = Insert(T[nid], obj_m.dis[data_idx + i + part_num_prefix[bid]],
								obj_m.res_id[data_idx + i + part_num_prefix[bid]], node_idx, tree_idx, T);
			}
		}
	}
}

__global__ void init(BPlusNode **NewNode, int num)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int total_num = gridDim.x * blockDim.x;

	for (int i = idx; i < num; i += total_num)
	{
		int m = 0;

		while (m < M + 1)
		{
			NewNode[i]->id[m] = Unavailable;
			NewNode[i]->Key[m] = Unavailable;
			NewNode[i]->Children[m] = NULL;
			m++;
		}
		NewNode[i]->Next = NULL;
		NewNode[i]->KeyNum[0] = 0;
		NewNode[i]->idx[0] = i;
	}
}

__global__ void destroy(BPlusNode **T, int num)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	for (int i = idx; i < num; i += blockDim.x * gridDim.x)
	{
		Destroy(T[i]);
	}
}

__global__ void show(BPlusNode *T)
{
	printf("show: %d\n", T->KeyNum[0]);
}

__device__ int getHeight(int m, int n)
{
	int height = 1;
	int max_leaf_node = (int)pow((float)m, height - 1);
	while (max_leaf_node < n)
	{
		height++;
		max_leaf_node = (int)pow((float)m, height - 1);
	}
	return height;
}

__device__ int getNodeNum(int m, int n)
{
	// int num_leaves = (n - 1) / (LIMIT_M_2) + 1;
	////printf("%d\n", num_leaves);
	// int h = getHeight(m, num_leaves);
	////printf("%d\n", h);
	// int num_internal_nodes = 0;
	// for (int i = 0; i < h - 1; i++) {
	//	num_internal_nodes += powf(LIMIT_M_2, i);
	// }
	// return (num_leaves + num_internal_nodes);
	return n;
}

// Calculate tree num of each partition
__global__ void getTreeNum(int pnum, int *part_num, int *tree_num)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int total_num = gridDim.x * blockDim.x;

	for (int idx = id; idx < pnum; idx += total_num)
	{
		tree_num[idx] = (part_num[idx] - 1) / DNUM + 1;
	}
}

__global__ void getNodeMax(int pnum, int *tree_num, int *part_num, int *tree_sum_prefix, int *node_num)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;

	if (bid < pnum)
	{
		for (int id = tid; id < tree_num[bid]; id += blockDim.x)
		{
			int num;
			if (id == tree_num[bid] - 1)
			{
				num = part_num[bid] - DNUM * (tree_num[bid] - 1);
			}
			else
			{
				num = DNUM;
			}
			int idx = tree_sum_prefix[bid] + id;
			node_num[idx] = getNodeNum(M, num);
			// printf("node num: %d, num: %d\n ", node_num[idx], num);
		}
	}
}

__global__ void sortPartData(int *res_id, float *dis, int *part_num, int *part_num_prefix)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int total_num = gridDim.x * blockDim.x;

	for (int idx = id; idx < PNUM; idx += total_num)
	{
		int start_idx = part_num_prefix[idx];
		int end_idx = start_idx + part_num[idx];
		thrust::sort_by_key(thrust::device, dis + start_idx, dis + end_idx, res_id + start_idx);
	}
}

// Construction of index
void getIndex(BPlusNode **&T, int *part_num, Obj obj_m, int &total_node_num, int *&tree_num, int *&tree_sum_prefix,
			  int &total_tree_num, int *&node_num, int *&node_sum_prefix)
{
	// cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1024 * 1024 * 1024);
	cout << "Index construction..." << endl;

	int *node_idx; // index of newly split node
	int *part_num_prefix;

	cudaMallocManaged((void **)&tree_num, PNUM * sizeof(int));
	cudaMallocManaged((void **)&tree_sum_prefix, PNUM * sizeof(int));
	cudaMalloc((void **)&part_num_prefix, PNUM * sizeof(int));

	thrust::exclusive_scan(thrust::device, part_num, part_num + PNUM, part_num_prefix); // in-place scan
	cudaDeviceSynchronize();

	int block_tree_num = (PNUM - 1) / THREAD_NUM + 1;
	/*sortPartData << <block_tree_num, THREAD_NUM >> > (obj_m.res_id, obj_m.dis, part_num, part_num_prefix);
	cudaDeviceSynchronize();*/

	// Calculate tree num of each partition
	getTreeNum<<<block_tree_num, THREAD_NUM>>>(PNUM, part_num, tree_num);
	cudaDeviceSynchronize();

	// Calculate perfix tree sum and total tree num
	thrust::exclusive_scan(thrust::device, tree_num, tree_num + PNUM, tree_sum_prefix); // in-place scan
	cudaDeviceSynchronize();
	total_tree_num = tree_sum_prefix[PNUM - 1] + (tree_num[PNUM - 1]);
	cout << "total tree num: " << total_tree_num << endl;

	cudaMallocManaged((void **)&node_num, total_tree_num * sizeof(int));
	cudaMallocManaged((void **)&node_sum_prefix, total_tree_num * sizeof(int));

	// Calculate max node num of each B+-tree
	getNodeMax<<<PNUM, THREAD_NUM>>>(PNUM, tree_num, part_num, tree_sum_prefix, node_num);
	cudaDeviceSynchronize();

	// Calculate perfix node sum and total node num
	thrust::exclusive_scan(thrust::device, node_num, node_num + total_tree_num, node_sum_prefix); // in-place scan
	cudaDeviceSynchronize();
	total_node_num = node_sum_prefix[total_tree_num - 1] + (node_num[total_tree_num - 1]);
	// cout << "total node num: " << total_node_num << endl;

	CHECK(cudaMallocManaged((void **)&T, total_node_num * sizeof(BPlusNode *)));
	for (int i = 0; i < total_node_num; i++)
	{
		CHECK(cudaMalloc((void **)&(T[i]), sizeof(BPlusNode)));
	}
	CHECK(cudaMalloc((void **)&(node_idx), total_tree_num * sizeof(int)));

	// Initialize each tree node
	int init_num = (total_node_num - 1) / THREAD_NUM + 1;
	init<<<init_num, THREAD_NUM>>>(T, total_node_num);
	cudaDeviceSynchronize();
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: %s\n", cudaGetErrorString(cudaStatus));
		// goto Error;
	}

	// Construction of B+-tree
	getTree<<<PNUM, THREAD_NUM>>>(T, part_num, tree_sum_prefix, obj_m, PNUM, tree_num, node_sum_prefix, node_idx, part_num_prefix);
	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: %s\n", cudaGetErrorString(cudaStatus));
		// goto Error;
	}

	int tree_idx = tree_sum_prefix[1];		 // Tree idx
	int node_id = node_sum_prefix[tree_idx]; // Node idx
	// cout << node_id << endl;

	// travelTree << <1, 1 >> > (T, node_id);
	// cudaDeviceSynchronize();
	// cudaStatus = cudaGetLastError();
	// if (cudaStatus != cudaSuccess) {

	//	fprintf(stderr, "Cuda error: %s\n", cudaGetErrorString(cudaStatus));
	//	//goto Error;
	//}

	cudaFree(node_idx);
	cudaFree(part_num_prefix);
}