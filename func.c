#include <stdlib.h>
#include <math.h>
#include "const.h"
#include <mpi/mpi.h>


/**
 * @brief initiates MPI and returns by address number of process and process' rank
 * @note Case number of processes different from what planned - it aborts the program
 * @param argc - arguments count
 * @param argv - arguments value
 * @param num_processes - pointer for returned value for number of processes
 * @param process_rank - pointer for returned value for process' rank
 */
void mpi_init(int argc, char* argv[], int* num_processes, int* process_rank)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, num_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, process_rank);

    if (*num_processes != NUM_OF_PROCESSES)
    {
        puts(PROCESS_ERROR_FORMAT);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

/**
 * @brief open file pointer with a specific format.
 * @note case this function fails - it aborts the program
 * @param file_name - a string contains the name of the wanted file
 * @param format - a string contains the format (read/write)
 * @return a pointer to the file variable
 */
FILE* my_file_open(const char* file_name, const char* format)
{
    FILE* f = fopen(file_name, format);
    if (!f)
    {
        puts(FILE_ERROR_FORMAT);
        MPI_Abort(MPI_COMM_WORLD, EXIT_CODE);
    }
    return f;
}

/**
 * @brief a malloc function that makes the test of failed allocation
 * @note case this function fails - it aborts the program
 * @param bytes_per_element - the number of bytes for each element
 * @param element_count - the total number of elements
 * @return a pointer for the allocated memory
 */
void* my_malloc(size_t bytes_per_element ,int element_count)
{
    void* p = NULL;
    p = (void*) malloc(bytes_per_element * element_count);
    if(!p)
    {
        puts("Allocation error");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return p;
}

/**
 * @brief reads line after line of a file and parsing values by a specific order
 * @param f - a pointer for the file
 * @param points_arr - an array of points
 * @param size - the size of the array
 */
void read_points_from_file(FILE* f, Point* points_arr, int size)
{
    int i;

    for(i = 0; i < size; ++i)
    {
        fscanf(f, "%d %lf %lf %lf %lf",
               &points_arr[i].Nid, &points_arr[i].x1, &points_arr[i].x2,
               &points_arr[i].a, &points_arr[i].b);
    }
}

/**
 * @brief Calculating all T values, and for each - check weather it fit for the problem definition
 * @note Each T affects all the points array, each T will force calculating all points data.
 * @param begin_index - index to start calculating T
 * @param end_index - index to finish calculating T
 * @param process_rank - which process is calculating
 * @param work_size - help adjusting the calculating of T value
 * @param TCount - help for calculating of T
 * @param points_arr - an array of points
 * @param N - the number of points
 * @param K - defined in the file
 * @param D - defined in the file
 * @param local_results - sub array of points to work on
 * @param index_local_results - pointer to index of the sub array of points
 */
void calculate_fit_ts(int begin_index, int end_index, int process_rank, int work_size,
                      int TCount, Point* points_arr, int N, int K, double D,
                      PointProCent* local_results, int* index_local_results)
{
    int i, j;
    double t;
    for (i = begin_index; i < end_index; ++i)
    {
        t = 2.0 * (i + (process_rank * work_size)) / TCount - 1;

#pragma omp parallel for private(j) shared(points_arr)
        for(j = 0; j < N; ++j)
        {
            points_arr[j].x = ((points_arr[j].x2 - points_arr[j].x1) / 2)
                              * sin(t * M_PI / 2) +
                              (points_arr[j].x2 + points_arr[j].x1) / 2;
            points_arr[j].y = points_arr[j].a * points_arr[j].x + points_arr[j].b;
        }

#pragma omp barrier // Wait for all threads to finish their jobs

        // Case T is valid - the sub array of points should keep the data
        if (compute_CUDA(points_arr, N, K, D, local_results, *index_local_results) == True)
        {
            local_results[*index_local_results].isValid = True;
            local_results[*index_local_results].t = t;
            *index_local_results += 1;
        }
    }

}