# GTS: GPU-based Tree Index for Fast Similarity Search

## Introduction

GTS is a GPU-based tree index designed for parallel processing of concurrent similarity search in general metric spaces, where only the distance metric for measuring object similarity is known. The GTS index utilizes a pivot based-tree structure for efficient object pruning and employs list tables to facilitate GPU computing. To efficiently handle concurrent similarity queries with limited GPU memory, a two-stage search method that combines the batch processing and  sequential strategies is developed to control memory usage. We also introduces an effective update strategy for GTS, covering streaming data updates and batch data  updates. 

## Development

We implement our index using C++ and NVIDIA CUDA (a platform and programming model for parallel computing on GPUs). We use CMake to compile and build the project. 

The code for this project is located in the "GTS" directory. The files in the "GTS" directory are organized as follows.

- include: It encompasses all the functionalities, including file reading (file.cuh), GTS construction (tree.cuh), similarity search (search.cuh), and updates (update.cuh and search_naive.cuh).
- src: It encompasses the main function file that constructs a GTS index based on input data and utilizes GTS for concurrent metric range queries and concurrent metric $k$NN queries.
- CMakeLists.txt

The steps for building the project are as follows:

```shell
cd GTS
mkdir bin
mkdir build
cd build 
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j
```

The compiled executable file "gpu_tree" is located in the "GTS/bin" directory.

## Examples

### Index Construction and Similarity Search

Build GTS index based on the input data, and utilize GTS for similarity search.

```shell
GTS/bin/GTS [data_path] [query_path] [operation_type] [operation_para] [result_path]
```

The meanings of the parameters are as follows.

- **data_path**, The file contains all the data, and the first line for datasets denotes the dimensionality, the size and the used distance metric.
- **query_path**, The file contains the IDs for the quries and the first line denotes the number of concurrent queries.
- **operation_type**, Type of operation, with 0 for $k$NN queries, 1 for range queries and 2 for updates.
- **operation_para**, Parameter for the corresponding operation, representing '$k$' in $k$NN queries or the radius in range queries. 
-  **result_path**, Result file path.

Here is a specific example for GTS index construction and similarity search.

```shell
cd GTS
bin/GTS ../Datasets/sbw_example.txt ../Datasets/sbw_example_qid.txt 0 8 cost_q_k.txt
```

### Index Updating

When confronted with **batch updates**, we choose to directly reconstruct the index. Leveraging our efficient parallel index construction method, performing a direct batch reconstruction of the modified dataset proves both feasible and effective.

When data arrives in a sequential streaming fashion (**stream data updates**), we propose to use lazy strategy with a cache list to handle streaming data updates.

The example of the stream data updates is as follows. Range queries are mixed In the process of data insertion and deletion.

```shell
GTS/bin/GTS [data_path] [update_path] [operation_type] [operation_para] [result_path]
```

The meanings of the parameters are as follows.

- **data_path**, The file contains all the data, and the first line for datasets denotes the dimensionality, the size and the used metric.
- **update_path**, The file contains the stream data and corresponding operations. The first line denotes the size. Afterwards, each line consists of an operation and a data ID. Specifically, for each operation, 0 represents an insertion, 1 represents a deletion, and 2 represents a query.
- **operation_type**, Type of operation, with 0 for $k$NN queries, 1 for range queries and 2 for updates.
- **operation_para**, Representing the radius in range queries when data arrives in a sequential streaming fashion.
- **result_path**, Result file path.

Here is a specific example for steam data updates.

```shell
cd GTS
bin/GTS ../Datasets/sbw_example.txt ../Datasets/sbw_example_u.txt 2 59 cost_u.txt
```


## Baselines

| __Algorithm__ | __Paper__ | __Year__ |
|-------------|------------|------------|
|BST   | A Data Structure and an Algorithm for the Nearest Point Problem | 1983 |
|EGNAT   | Searching and Updating Metric Space Databases Using the Parallel EGNAT | 2007 |
|MVPT | Distance-Based Indexing for High-Dimensional Metric Spaces | 1997 |
|GPU-Table | - GPU-accelerated table-based methods | - |
|GPU-Tree | - GPU-accelerated tree-based methods | - |
|LBPG-Tree | Multi-GPU Efficient Indexing For Maximizing Parallelism of High Dimensional Range Query Services | 2022 |
|GANNS | GPU-accelerated Proximity Graph Approximate Nearest Neighbor Search and Construction | 2022 |

- We choose the CPU baselines according to the Chen et al.'s survey in CSUR 2022, where BST, EGNAT, and MVPT show superiority performance among all CPU-based approaches for exact similarity search. The source codes for BST, EGNAT and MVPT are available at the [SISAP Metric Space Library](https://www.sisap.org/).


- GPU-Table computes the distances between query and all the objects to answer MRQ and leverage [Dr. Top-$k$ algorithm](https://dl.acm.org/doi/pdf/10.1145/3458817.3476141) to answer M$k$NNQ. 


- GPU-Tree implements the SOTA GPU-based tree index [G-PICS](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9086127) strategy for general similarity search on single GPU by constructing multiple MVP-trees. The source code of our implementation for GPU-Tree is in the "GPU-Tree" folder, and its compilation and similarity search example are performed similarly to GTS. Here is a specific example for GPU-Tree index construction and similarity search.

```shell
cd GPU-Tree
bin/gpu_tree ../Datasets/sbw_example.txt ../Datasets/sbw_example_qid.txt 0 8
```

- [LBPG-Tree](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9430517) constructs R-Trees on GPU high dimensional range query services. Notably, we also implement the G-PICS's  $k$ nearest neighbour search strategy for LBPG-Tree to support M$k$NNQ.

- [GANNS](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9835618) is a GPU-based graph method for vector similarity search. 

## Datasets

Each dataset can be obtained from the following links. 

| Dataset | Cardinality | Dimensionality | Distance Metric      | Link                                          |
| ------- | ----------- | -------------- | -------------------- | --------------------------------------------- |
| Words   | 611,756     | 1∼34           | Edit distance        | https://mobyproject.org                       |
| T-Loc   | 10,000,000  | 2              | $L_2$-norm           | http://www.vldb.org/pvldb/vol12/p99-ghosh.pdf |
| Vector  | 200,000     | 300            | Word cosine distance | https://code.google.com/archive/p/word2vec    |
| DNA     | 1,000,000   | 108            | Edit distance        | http://www.ncbi.nlm.nih.gov/genome            |
| Color   | 5,000,000   | 282            | $L_1$-norm           | http://cophir.isti.cnr.it                     |

The dataset file is in txt format. The example of dataset file is located in "Datasets Sample". 
