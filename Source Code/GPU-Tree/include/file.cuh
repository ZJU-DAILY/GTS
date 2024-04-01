// FIle operations
// Created on 24-01-05

#pragma once
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <stdio.h>
#include <iostream>
#include "config.cuh"
using namespace std;
#define MaxC 109

// Result struct
typedef struct Object
{
	float *dis;	 // distance
	int *res_id; // result id
	int *flag;	 // flag for data partition
} Obj;

// Split the line
void split(const string str, vector<string> &res, const char pattern)
{
	istringstream is(str);
	string temp;
	while (getline(is, temp, pattern))
		res.push_back(temp);
	return;
}

// Load data file
void load(char *file, int *&data_info, short *&data_d, char *&data_s, int *&size_s)
{
	ifstream in(file);
	if (!in.is_open())
	{
		std::cout << "open file error" << std::endl;
		exit(-1);
	}

	cout << "Loading data file..." << endl;

	string line;
	int i = 0;
	int j = 0;
	vector<string> res;

	cudaMallocManaged((void **)&data_info, 3 * sizeof(int));

	// Load the file
	while (getline(in, line))
	{
		if (i == 0)
		{ // Load the first line
			split(line, res, ' ');
			for (auto r : res)
			{
				stringstream ss(r);
				int number;
				ss >> number;

				if (j == 0)
					data_info[j] = number;
				if (j == 1)
					data_info[j] = number;
				if (j == 2)
					data_info[j] = number;

				j++;
			}

			// Allocate memory
			if (data_info[2] != 6)
			{
				cudaMallocManaged((void **)&data_d, data_info[1] * data_info[0] * sizeof(short));
			}
			else
			{
				cudaMallocManaged((void **)&data_s, data_info[1] * MaxC * sizeof(char));
				cudaMallocManaged((void **)&size_s, data_info[1] * sizeof(int));
			}
		}
		else
		{ // Load data
			if (data_info[2] != 6)
			{ // float
				split(line, res, ' ');
				for (auto r : res)
				{
					stringstream ss(r);
					float number;
					ss >> number;

					*(data_d + (i - 1) * data_info[0] + j) = number;

					j++;
				}
			}
			else
			{ // string
				const char *temp = line.data();
				memcpy(data_s + (i - 1) * MaxC, temp, strlen(temp));
				size_s[i - 1] = strlen(temp);
			}
		}

		res.clear();
		j = 0;
		i++;
	}

	in.close();
}

// Load query file
void loadQuery(char *file, int *&qid, int &qnum)
{
	ifstream in(file);
	if (!in.is_open())
	{
		std::cout << "open file error" << std::endl;
		exit(-1);
	}

	cout << "Loading query file..." << endl;

	string line;
	int i = 0;

	// load the file
	while (getline(in, line))
	{
		if (i == 0)
		{ // load the first line
			stringstream ss(line);
			int number;
			ss >> number;

			cudaMallocManaged((void **)&qid, number * sizeof(int));
			qnum = number;
		}
		else
		{ // load query id
			stringstream ss(line);
			int number;
			ss >> number;

			qid[i - 1] = number;
		}

		i++;
	}

	in.close();
}

// Save result file for knn
void saveK(char *filenamer, char *filenamed, int *res_id, float *dis, int k, int qnum)
{
	FILE *resr = fopen(filenamer, "w");
	FILE *resd = fopen(filenamed, "w");

	for (unsigned i = 0; i < qnum; i++)
	{
		for (unsigned j = 0; j < k; j++)
		{
			fprintf(resr, "%d ", res_id[i * k + j]);
			fflush(resr);
			fprintf(resd, "%f ", dis[i * k + j]);
			fflush(resd);
		}
		fprintf(resr, "\n");
		fflush(resr);
		fprintf(resd, "\n");
		fflush(resd);
	}
}

// Save result file for rnn
void saveR(char *filenamer, char *filenamed, int qnum, Obj init_result, int *result_counter, int *qnum_counter, int *qnum_counter_prefix, int DNUM)
{
	FILE *resr = fopen(filenamer, "w");
	FILE *resd = fopen(filenamed, "w");

	for (int k = 0; k < qnum; k++)
	{
		for (int i = 0; i < qnum_counter[k]; i++)
		{
			int id = qnum_counter_prefix[k] + i;
			for (int j = 0; j < result_counter[id]; j++)
			{
				int idx = j + id * DNUM;
				fprintf(resr, "%d ", init_result.res_id[idx]);
				fflush(resr);
				fprintf(resd, "%f ", init_result.dis[idx]);
				fflush(resr);
			}
		}
		fprintf(resr, "\n");
		fflush(resr);
		fprintf(resd, "\n");
		fflush(resd);
	}

	fclose(resr);
	fclose(resd);
}
