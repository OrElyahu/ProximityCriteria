#include <stdio.h>
#include <stdlib.h>
#include "const.h"
#include <omp.h>
#include <mpi/mpi.h>


int main(int argc, char* argv[])
{
    int num_processes, process_rank, N, K, TCount, i,work_size, index_local_results, reminder;
    double D;
    FILE* f = NULL;
    Point* points_arr = NULL;
    PointProCent *results = NULL , *local_results = NULL;
    Boolean found = False;
    index_local_results = 0;

    mpi_init(argc, argv, &num_processes, &process_rank);

    // Let master read first data and assign results
    if (process_rank == MASTER_RANK)
    {
        f = my_file_open(FILE_INPUT, READ_FORMAT);
        fscanf(f, "%d %d %lf %d", &N, &K, &D, &TCount);
        results = (PointProCent*) my_malloc(sizeof(PointProCent), (TCount + 1));

        //By default, results is shared
        #pragma omp parallel for
        for (i = 0; i < TCount + 1; ++i)
        {
            results[i].isValid = False;
        }
    }

    // Transfer data from Master to all
    MPI_Bcast(&N, 1, MPI_INT, MASTER_RANK, MPI_COMM_WORLD);
    MPI_Bcast(&K, 1, MPI_INT, MASTER_RANK, MPI_COMM_WORLD);
    MPI_Bcast(&D, 1, MPI_DOUBLE, MASTER_RANK, MPI_COMM_WORLD);
    MPI_Bcast(&TCount, 1, MPI_INT, MASTER_RANK, MPI_COMM_WORLD);

    // Define every process jobs
    work_size = (TCount + 1) / num_processes;
    points_arr = (Point*) my_malloc(sizeof(Point), N);
    local_results = (PointProCent*) my_malloc(sizeof(PointProCent), work_size);
    reminder = (TCount + 1) % work_size;

    // Let master read all points data
    if (process_rank == MASTER_RANK)
    {
        read_points_from_file(f,points_arr,N);
        fclose(f);
    }

    // Transfer points from Master to all
    MPI_Bcast(points_arr, (int)(N * sizeof(Point)),
              MPI_BYTE, MASTER_RANK, MPI_COMM_WORLD);

    //Each process will use the max threads, and yet not making time slicing
    omp_set_num_threads(MIN_THREADS_NUM);

    // Each process works on a subarray of T's
    calculate_fit_ts(0, work_size, process_rank, work_size, TCount,
                     points_arr, N, K, D, local_results, &index_local_results);

    // Each process sends subarray to Master
    MPI_Gather(local_results, (int)(work_size * sizeof(PointProCent)) , MPI_BYTE ,
               results, (int)(work_size * sizeof(PointProCent)), MPI_BYTE,
               MASTER_RANK, MPI_COMM_WORLD);


    if(process_rank == MASTER_RANK)
    {
        // Master will use all threads, all process finish their jobs
        omp_set_num_threads(MAX_THREADS_NUM);
        //handle reminder with Master
        if(reminder != 0)
        {
            index_local_results = work_size * num_processes;
            calculate_fit_ts(work_size,work_size + reminder, num_processes - 1,
                             work_size, TCount,
                             points_arr, N, K, D, results, &index_local_results);
        }
        f = my_file_open(FILE_OUTPUT, WRITE_FORMAT);

        #pragma omp parallel for shared(results, found)
        for (i = 0; i <= TCount; ++i)
        {
            if (results[i].isValid == True)
            {
                #pragma omp critical
                {
                    found = True;
                    fprintf(f,
                            FOUND_POINT_FORMAT,
                            results[i].points_index[0], results[i].points_index[1], results[i].points_index[2], results[i].t);
                }
            }
        }

        #pragma omp barrier // Wait for all threads to finish their jobs
        if (found == False)
        {
            fprintf(f, NOT_FOUND_POINTS_FORMAT);
        }

        free(results);
        fclose(f);
    }

    //Free allocations
    free(points_arr);
    free(local_results);

    MPI_Finalize();

    return 0;
}
