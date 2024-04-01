// GTS index construction and similarity search with GTS
// Created on 24-01-05

#include <chrono>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include "tree.cuh"
#include "file.cuh"
#include "search.cuh"
#include "update.cuh"
#include "search_v2.cuh"
#include "config.cuh"

int *data_info;
short *data_d;
char *data_s;
int *size_s;
int *qid_list;
int qnum;
int *max_node_num;
int *id_list;
TN *node_list;
char *file;
char *file_q;
char *file_u;
float time_index = 0;
float time_search = 0;
float time_update_s = 0;
float time_update_u = 0;
int count_update_s = 0;
int count_update_u = 0;
int tree_h;
int k;	 // k for knn
float r; // r for range query
int *empty_list;
int *qresult_count;
int *qresult_count_prefix;
int *result_id;
float *result_dis;
int process_type;
int search_type;

int main(int argc, char **argv)
{
	file = argv[1];
	load(file, data_info, data_d, data_s, size_s);
	process_type = (int)atoi(argv[3]);
	if (process_type != 2)
	{
		file_q = argv[2];
		loadQuery(file_q, qid_list, qnum);
		k = (int)atoi(argv[4]);
		r = (float)stod(argv[4]);
		// TREE_ORDER = (int)atoi(argv[5]);
		// MAX_SIZE = (int)atoi(argv[6]);
		// MAX_H = (int)atoi(argv[7]);
		// DIS_CODE = (int)atoi(argv[8]);
		// INFI_DIS = (int)atoi(argv[9]);
		// float temp_s = (float)stod(argv[10]);
		// input_size = temp_s * 1024 * 1024 * 1024;
		// printf("%f, %f\n", temp_s, input_size);
	}
	else
	{
		file_u = argv[2];
		loadUpdate(file_u, update_list, update_num);
		// search_type = (int)atoi(argv[4]);
		search_type = 1;
		// k = (int)atoi(argv[5]);
		r = (float)stod(argv[4]);
		// TREE_ORDER = (int)atoi(argv[6]);
		// MAX_SIZE = (int)atoi(argv[7]);
		// MAX_H = (int)atoi(argv[8]);
		// DIS_CODE = (int)atoi(argv[9]);
		// INFI_DIS = (int)atoi(argv[10]);
		// MAX_IN_SIZE = (int)atoi(argv[11]);
	}

	// Index Construction
	auto s = std::chrono::high_resolution_clock::now();
	indexConstru(data_d, data_s, size_s, data_info, id_list, node_list, max_node_num, tree_h, empty_list);
	auto e = std::chrono::high_resolution_clock::now();
	std::chrono::duration<float> diff = e - s;
	time_index += diff.count();

	// knn
	if (process_type == 0)
	{
		FILE *fcost = fopen(argv[5], "w");
		fprintf(fcost, "Knn search num: %d\nResult radius: \n", k);
		fflush(fcost);

		// knn
		s = std::chrono::high_resolution_clock::now();
		searchIndexKnnV2(data_d, node_list, id_list, max_node_num, qid_list, qnum, k, tree_h, data_info, empty_list, data_s, size_s);
		e = std::chrono::high_resolution_clock::now();
		diff = e - s;
		time_search += diff.count();

		// Output results
		for (int i = 0; i < qnum; i++)
		{
			fprintf(fcost, "%f ", res_dis[i]);
			fflush(fcost);
		}
		printf("Time of index construction: %f\n", time_index);
		printf("Search time: %f\n", time_search / qnum);
		fprintf(fcost, "\nTime of index construction: %f\n", time_index);
		fprintf(fcost, "Search time: %f\n", time_search / qnum);
		fflush(fcost);
		fclose(fcost);
	}

	// Range query
	else if (process_type == 1)
	{
		FILE *fcost = fopen(argv[5], "w");
		fprintf(fcost, "Range search radius: %f\nResult num: \n", r);
		fflush(fcost);

		// Range query
		s = std::chrono::high_resolution_clock::now();
		searchIndexRnnV2(data_d, node_list, id_list, max_node_num, qid_list, qnum, r, tree_h, data_info, empty_list, data_s, size_s);
		e = std::chrono::high_resolution_clock::now();
		diff = e - s;
		time_search += diff.count();

		// Output results
		for (int i = 0; i < qnum; i++)
		{
			fprintf(fcost, "%d ", res[i]);
			fflush(fcost);
		}
		printf("\nTime of index construction: %f\n", time_index);
		printf("Search time: %f\n", time_search / qnum);
		fprintf(fcost, "\nTime of index construction: %f\n", time_index);
		fprintf(fcost, "Search time: %f\n", time_search / qnum);
		fflush(fcost);
		fclose(fcost);
	}

	// Update
	else
	{
		FILE *fcost = fopen(argv[5], "w");
		fprintf(fcost, "Range search radius (check for updates): %f\nResult num: \n", r);
		fflush(fcost);

		// Update
		updateIndexRnn(data_d, node_list, id_list, max_node_num, qid_list, 1, r, tree_h, data_info, empty_list,
					   qresult_count, qresult_count_prefix, result_id, result_dis, data_s, size_s, fcost, time_update_s, time_update_u,
					   count_update_s, count_update_u);

		// Output results
		printf("Time of index construction: %f\n", time_index);
		printf("Total update time: %f\n", time_update_s / count_update_s + time_update_u / count_update_u);
		printf("Search time in update: %f\n", time_update_s / count_update_s);
		printf("Update time in update: %f\n", time_update_u / count_update_u);
		fprintf(fcost, "\nTime of index construction: %f\n", time_index);
		fprintf(fcost, "Total update time: %f\n", time_update_s / count_update_s + time_update_u / count_update_u);
		fprintf(fcost, "Search time in update : % f\n", time_update_s / count_update_s);
		fprintf(fcost, "Update time in update: %f\n", time_update_u / count_update_u);
		fflush(fcost);
		fclose(fcost);
	}

	// Release memory
	cudaFree(data_info);
	cudaFree(data_d);
	cudaFree(data_s);
	cudaFree(size_s);
	cudaFree(id_list);
	cudaFree(node_list);
	cudaFree(max_node_num);
	cudaFree(qid_list);
	cudaFree(empty_list);
	cudaFree(update_list);
	cudaFree(res);
	cudaFree(res_dis);
	return 0;
}