#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <omp.h>
#include "const.h"


/**
 * @note This is function called from GPU and executed on GPU
 * @param p - first point
 * @param p2 - second point
 * @return the distance between two points
 */
__device__ double kernel_distance(const Point* p, const Point* p2)
{
    double x_square, y_square, res;
    x_square = pow(p->x - p2->x, 2);
    y_square = pow(p->y - p2->y, 2);
    res = sqrt(x_square + y_square);
    return  res;
}

/**
 * @brief This function calculating weather there are K points with distance greater than D
 * @note This function called from CPU and executed on GPU
 * @note function using atomic add to avoid adding at the same time
 * @param points_arr - an arr of points
 * @param N - arr size
 * @param K - number of points that supposed to be in a distance greater than D
 * @param D - distance to be greater than
 * @param point_index - point index to find if there are K different points with distance greater than D
 * @param points_range_count - pointer for returned value, weather there are or aren't K points
 */
__global__ void kernel_function(const Point* points_arr, int N, int K, double D,
                                int point_index, int* points_range_count)
{
    int i  = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N && i != point_index && *points_range_count < K)
    {
        if (kernel_distance(&points_arr[point_index], &points_arr[i]) < D)
        {
            atomicAdd(points_range_count,1);
        }
    }
}

/**
 * @brief a malloc for cuda function that makes the test of failed allocation
 * @note case this function fails - it aborts the program
 * @param size - number of bytes to be malloced on GPU
 * @return a pointer for the malloced memory on the GPU
 */
void* my_cuda_malloc(size_t size)
{
    void* ptr = NULL;
    cudaError_t err;

    err = cudaMalloc((void **)&ptr, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    return ptr;
}

/**
 * @brief a memory copy for cuda function that makes the test of failed allocation
 * @note case this function fails - it aborts the program
 * @param dst - a pointer for the destination
 * @param src - a pointer for the source
 * @param count - how many bytes to be copied
 * @param kind - weather it's from host to device or device to host
 */
void my_cuda_mem_cpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind)
{
    cudaError_t err;
    err = cudaMemcpy(dst, src, count, kind);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy data from %s - %s\n",
                (kind == cudaMemcpyHostToDevice) ? "host to device" : "device to host",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

/**
 * @brief a free memory cuda function that makes the test of failed allocation
 * @note case this function fails - it aborts the program
 * @param ptr - the pointer to be free on the GPU
 */
void my_cuda_free(void* ptr)
{
    cudaError_t err;
    err = cudaFree(ptr);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device data - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

/**
 * @brief this function validate weather there are 3 points that satisfied problem definition using cuda
 * @note This function using GPU implementation. Case something goes wrong - it aborts the program
 * @param points_arr - an array of points
 * @param N - the size of the array
 * @param K - number of points that supposed to be in a distance greater than D
 * @param D - distance to be greater than
 * @param local_results - sub array of desired points (K different points greater than distance D)
 * @param index_local_results - index of the local_results array
 * @return True if there are at least SATISFIED_IDS_SIZE that fit the problem definition. False otherwise.
 */
Boolean compute_CUDA(const Point *points_arr, int N, int K, double D,
                     PointProCent* local_results, int index_local_results)
{
    int i, points_range_count, points_satisfied_pro_count, threadsPerBlock, blocksPerGrid;
    int* d_points_range_count;
    size_t arr_size;
    Point *cuda_arr = NULL;
    cudaError_t err;

    // Define GPU variables
    arr_size = N * sizeof(Point);
    cuda_arr = (Point *) my_cuda_malloc(arr_size);
    d_points_range_count = (int*) my_cuda_malloc(sizeof(int));
    my_cuda_mem_cpy(cuda_arr, points_arr, arr_size, cudaMemcpyHostToDevice);

    // Define the number of threads to work simultaneously
    threadsPerBlock = MAX_THREADS_PER_BLOCK;
    blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    points_satisfied_pro_count = 0;

    for(i = 0; i < N && points_satisfied_pro_count < SATISFIED_IDS_SIZE; ++i)
    {
        // Set memory to GPU
        points_range_count = 0;
        my_cuda_mem_cpy(d_points_range_count, &points_range_count,
                        sizeof(int), cudaMemcpyHostToDevice);

        kernel_function<<<blocksPerGrid, threadsPerBlock>>>
                (cuda_arr, N, K, D, i,d_points_range_count);

        // Validate kernel function worked properly
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, ERROR_KERNEL_FORMAT, cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        // Get results from GPU
        my_cuda_mem_cpy(&points_range_count, d_points_range_count,
                        sizeof(int), cudaMemcpyDeviceToHost);

        if(points_range_count >= K)
        {
            local_results[index_local_results].points_index[points_satisfied_pro_count++] = points_arr[i].Nid;
        }

    }

    // Free allocated memory on GPU
    my_cuda_free(cuda_arr);
    my_cuda_free(d_points_range_count);

    return (points_satisfied_pro_count >= SATISFIED_IDS_SIZE) ? True : False;

}