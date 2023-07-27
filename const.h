#ifndef CONST_H
#define CONST_H

#include <stdio.h>
#include <stdlib.h>


// Strings
#define FILE_INPUT "Input.txt"
#define FILE_OUTPUT "Output.txt"
#define READ_FORMAT "r"
#define WRITE_FORMAT "w"
#define FILE_ERROR_FORMAT "Error on opening file."
#define PROCESS_ERROR_FORMAT "This program should be run with 4 processes."
#define FOUND_POINT_FORMAT "Points pointID%d, pointID%d, pointID%d satisfy Proximity Criteria at t = %lf\n"
#define NOT_FOUND_POINTS_FORMAT "There were no 3 points found for any t.\n"
#define ERROR_KERNEL_FORMAT "Failed to launch function kernel -  %s\n"


// Ints
#define MASTER_RANK 0
#define EXIT_CODE 1
#define SATISFIED_IDS_SIZE 3
#define NUM_OF_PROCESSES 4
#define MAX_THREADS_PER_BLOCK 256
#define MIN_THREADS_NUM 2
#define MAX_THREADS_NUM 8

typedef enum {False, True} Boolean;

typedef struct{
    double x,y;
    double x2,x1,a,b;
    int Nid;
}Point;

typedef struct {
    int points_index[SATISFIED_IDS_SIZE];
    Boolean isValid;
    double t;
}PointProCent;

// On CPU
void mpi_init(int argc, char* [], int*, int*);
FILE* my_file_open(const char*, const char*);
void* my_malloc(size_t ,int);
void read_points_from_file(FILE*, Point*, int);
void calculate_fit_ts(int begin_index, int end_index, int process_rank, int work_size,
                      int TCount, Point* points_arr, int N, int K, double D,
                      PointProCent* local_results, int* index_local_results);

// On GPU
Boolean compute_CUDA(const Point *points_arr, int N, int K, double D, PointProCent* local_results, int index_local_results);
void* my_cuda_malloc(size_t);

#endif