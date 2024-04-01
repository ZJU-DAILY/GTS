// Queue
// Created on 24-01-05

#pragma once
#include "bplus_tree.cuh"
#include "config.cuh"

using namespace std;

#define MAXSIZE 500
typedef BPlusNode *DataType;

typedef struct Queue
{
	DataType queue[MAXSIZE];
	int front;
	int rear;
} SeqQueue;

__device__ void InitQueue(SeqQueue *SQ)
{
	if (!SQ)
		return;

	SQ->front = 0;
	SQ->rear = 0;
}

__device__ int IsEmpty(SeqQueue *SQ)
{
	if (!SQ)
		return 0;

	if (SQ->front == SQ->rear)
	{
		return 1;
	}

	return 0;
}

__device__ int IsFull(SeqQueue *SQ)
{
	if (!SQ)
		return 0;

	if ((SQ->rear + 1) % MAXSIZE == SQ->front)
	{
		return 1;
	}

	return 0;
}

__device__ void EnterQueue(SeqQueue *SQ, DataType data)
{
	if (!SQ)
	{
		printf("Error: The queue is NULL!!!\n");
		return;
	}

	if (IsFull(SQ))
	{
		printf("Error: The queue is full!!!\n");
		return;
	}

	SQ->queue[SQ->rear] = data;
	SQ->rear = (SQ->rear + 1) % MAXSIZE;
}

__device__ void DeleteQueeue(SeqQueue *SQ, DataType &data)
{
	if (!SQ || IsEmpty(SQ))
	{
		printf("Error: The queue is empty or NULL!!!\n");
		return;
	}

	data = SQ->queue[SQ->front];
	SQ->front = (SQ->front + 1) % MAXSIZE;
}

__device__ void GetHead(SeqQueue *SQ, DataType &data)
{
	if (!SQ || IsEmpty(SQ))
	{
		printf("Error: The queue is empty or NULL!!!\n");
		return;
	}

	data = SQ->queue[SQ->front];
}

__device__ int getLength(SeqQueue *SQ)
{
	if (!SQ)
		return 0;

	return (SQ->rear - SQ->front + MAXSIZE) % MAXSIZE;
}
