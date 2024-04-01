// GPU B+-tree method
// Created on 24-01-05

#include "search.cuh"
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <chrono>
#include <iostream>
#include "config.cuh"
using namespace std;

int main(int argc, char **argv)
{
	int *data_info;						  // dim num type
	short *data_d;						  // numeric type data
	char *data_s;						  // text type data
	int *size_s;						  // length of text
	int *qid;							  // query id
	int qnum;							  // query num
	char *file = argv[1];				  // data file
	char *file_q = argv[2];				  // query data file
	int *pid;							  // pivot id
	float *radius;						  // radius of each partition
	Obj obj_p;							  // whole result for data partition
	float time_p = 0;					  // time for getting pivots
	Obj obj_m;							  // part result for data partition
	int *part_num;						  // number of data in each partition
	BPlusNode **T;						  // node list of B+-tree
	float *dis_pivot;					  // distances between queries and pivots
	int total_node_num;					  // total tree node num
	int *isSatisfied;					  // flag of whether the partition has been pruned
	int *tree_num;						  // tree num of each partition
	int *tree_sum_prefix;				  // perfix tree sum of partitions
	int total_tree_num;					  // total tree num
	int *node_num;						  // max node num of each B+-tree
	int *node_sum_prefix;				  // prefix node sum of B+-trees
	int *qnum_counter;					  // search tree num of each query
	int *qnum_counter_prefix;			  // search tree num prefix of each query
	int *result_counter;				  // result num of each tree
	Obj init_result;					  // result for rnn
	float *dis_knn;						  // dis result for knn
	int *id_knn;						  // id result for knn
	int search_type = (int)atoi(argv[3]); // search type: 0 for knn; 1 for rnn
	int k = (int)atoi(argv[4]);			  // k for knn
	float r = (float)stod(argv[4]);		  // r for rnn
	// PNUM = (int)atoi(argv[5]);
	float time_dp = 0;
	float time_index = 0;
	float time_search = 0;

	cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1024 * 1024 * 1024);

	load(file, data_info, data_d, data_s, size_s);
	loadQuery(file_q, qid, qnum);

	cudaMallocManaged((void **)&pid, PNUM * sizeof(int));
	cudaMalloc((void **)&obj_p.dis, data_info[1] * sizeof(float));
	cudaMalloc((void **)&obj_p.res_id, data_info[1] * sizeof(int));
	cudaMalloc((void **)&obj_p.flag, data_info[1] * sizeof(int));
	cudaMalloc((void **)&part_num, PNUM * sizeof(int));
	cudaMalloc((void **)&obj_m.res_id, data_info[1] * sizeof(int));
	cudaMalloc((void **)&obj_m.dis, data_info[1] * sizeof(float));
	cudaMalloc((void **)&radius, PNUM * sizeof(float));
	cudaMalloc((void **)&dis_pivot, PNUM * qnum * sizeof(float));
	cudaMalloc((void **)&isSatisfied, PNUM * qnum * sizeof(int));
	// cudaMallocManaged((void**)&node_num, PNUM * sizeof(int));

	// Get pivot
	auto s = std::chrono::high_resolution_clock::now();
	getPivot(data_d, data_info, pid, PNUM, obj_p, data_s, size_s);
	auto e = std::chrono::high_resolution_clock::now();
	std::chrono::duration<float> diff = e - s;
	time_p += diff.count();

	// Get partition
	s = std::chrono::high_resolution_clock::now();
	getPartition(data_info[1], obj_m, obj_p, PNUM, part_num, radius);
	e = std::chrono::high_resolution_clock::now();
	diff = e - s;
	time_dp += diff.count();

	cudaFree(obj_p.dis);
	cudaFree(obj_p.res_id);
	cudaFree(obj_p.flag);

	// Get index
	s = std::chrono::high_resolution_clock::now();
	getIndex(T, part_num, obj_m, total_node_num, tree_num, tree_sum_prefix, total_tree_num, node_num, node_sum_prefix);
	e = std::chrono::high_resolution_clock::now();
	diff = e - s;
	time_index += diff.count();
	time_index = time_index + time_dp + time_p;

	s = std::chrono::high_resolution_clock::now();
	search(data_d, pid, qid, qnum, dis_pivot, data_info, r, radius, isSatisfied, tree_num, T,
		   tree_sum_prefix, node_sum_prefix, qnum_counter, qnum_counter_prefix, result_counter, init_result, search_type, k, dis_knn, id_knn,
		   data_s, size_s);
	e = std::chrono::high_resolution_clock::now();
	diff = e - s;
	time_search += diff.count();

	cout << "Time of getting pivots: " << time_p << "s" << endl;
	cout << "Time of data partition: " << time_dp << "s" << endl;
	cout << "Time of index construction: " << time_index << "s" << endl;
	cout << "Seach time: " << time_search / qnum << "s" << endl;

	cudaFree(data_d);
	cudaFree(qid);
	cudaFree(data_info);
	cudaFree(data_s);
	cudaFree(size_s);
	cudaFree(pid);

	cudaFree(obj_m.res_id);
	cudaFree(obj_m.dis);

	for (int i = 0; i < total_node_num; i++)
	{
		cudaFree(T[i]);
	}
	cudaFree(T);
	cudaFree(radius);
	cudaFree(dis_pivot);
	cudaFree(qnum_counter);
	cudaFree(qnum_counter_prefix);
	cudaFree(node_num);
	cudaFree(result_counter);
	cudaFree(init_result.dis);
	cudaFree(init_result.res_id);
	cudaFree(dis_knn);
	cudaFree(id_knn);

	return 0;
}