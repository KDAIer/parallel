#define _POSIX_C_SOURCE 199309L
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include "parallel_for.h"

#define M 500
#define N 500

typedef struct
{
    double u[M][N];
    double w[M][N];
    double diff;
    double my_diff;
    double mean;
    pthread_mutex_t mutex;
} thread_data_t;

int main(int argc, char *argv[]);
void set_top_boundary(int i, void *args);
void set_bottom_boundary(int i, void *args);
void set_left_boundary(int i, void *args);
void set_right_boundary(int i, void *args);
void set_initial_mean(int i, void *args);
void compute_new_solution(int i, void *args);
void compute_diff(int i, void *args);
void copy_solution(int i, void *args);

/******************************************************************************/

void set_top_boundary(int i, void *args)
{
    thread_data_t *data = (thread_data_t *)args;
    data->w[0][i] = 0.0;
}
void set_bottom_boundary(int i, void *args)
{
    thread_data_t *data = (thread_data_t *)args;
    data->w[M - 1][i] = 100.0;
}
void set_left_boundary(int i, void *args)
{
    thread_data_t *data = (thread_data_t *)args;
    data->w[i][0] = 100.0;
}
void set_right_boundary(int i, void *args)
{
    thread_data_t *data = (thread_data_t *)args;
    data->w[i][N - 1] = 100.0;
}
void set_initial_mean(int i, void *args)
{
    thread_data_t *data = (thread_data_t *)args;
    for (int j = 1; j < N - 1; j++)
    {
        data->w[i][j] = data->mean;
    }
}
void compute_new_solution(int i, void *args)
{
    thread_data_t *data = (thread_data_t *)args;
    for (int j = 1; j < N - 1; j++)
    {
        data->w[i][j] = (data->u[i - 1][j] + data->u[i + 1][j] + data->u[i][j - 1] + data->u[i][j + 1]) / 4.0;
    }
}
void compute_diff(int i, void *args)
{
    thread_data_t *data = (thread_data_t *)args;
    double local_diff = 0.0;
    for (int j = 1; j < N - 1; j++)
    {
        if (local_diff < fabs(data->w[i][j] - data->u[i][j]))
        {
            local_diff = fabs(data->w[i][j] - data->u[i][j]);
        }
    }
    pthread_mutex_lock(&data->mutex);
    if (data->diff < local_diff)
    {
        data->diff = local_diff;
    }
    pthread_mutex_unlock(&data->mutex);
}
void copy_solution(int i, void *args)
{
    thread_data_t *data = (thread_data_t *)args;
    for (int j = 0; j < N; j++)
    {
        data->u[i][j] = data->w[i][j];
    }
}

/******************************************************************************/

int main(int argc, char *argv[])

/******************************************************************************/
/*
  Purpose:

    MAIN is the main program for HEATED_PLATE_Pthread.

  Discussion:

    This code solves the steady state heat equation on a rectangular region.

    The sequential version of this program needs approximately
    18/epsilon iterations to complete.


    The physical region, and the boundary conditions, are suggested
    by this diagram;

                   W = 0
             +------------------+
             |                  |
    W = 100  |                  | W = 100
             |                  |
             +------------------+
                   W = 100

    The region is covered with a grid of M by N nodes, and an N by N
    array W is used to record the temperature.  The correspondence between
    array indices and locations in the region is suggested by giving the
    indices of the four corners:

                  I = 0
          [0][0]-------------[0][N-1]
             |                  |
      J = 0  |                  |  J = N-1
             |                  |
        [M-1][0]-----------[M-1][N-1]
                  I = M-1

    The steady state solution to the discrete heat equation satisfies the
    following condition at an interior grid point:

      W[Central] = (1/4) * ( W[North] + W[South] + W[East] + W[West] )

    where "Central" is the index of the grid point, "North" is the index
    of its immediate neighbor to the "north", and so on.

    Given an approximate solution of the steady state heat equation, a
    "better" solution is given by replacing each interior point by the
    average of its 4 neighbors - in other words, by using the condition
    as an ASSIGNMENT statement:

      W[Central]  <=  (1/4) * ( W[North] + W[South] + W[East] + W[West] )

    If this process is repeated often enough, the difference between successive
    estimates of the solution will go to zero.

    This program carries out such an iteration, using a tolerance specified by
    the user, and writes the final estimate of the solution to a file that can
    be used for graphic processing.

  Licensing:

    This code is distributed under the MIT license.

  Modified:

    18 October 2011

  Author:

    Original C version by Michael Quinn.
    This C version by John Burkardt.

  Reference:

    Michael Quinn,
    Parallel Programming in C with MPI and Pthread,
    McGraw-Hill, 2004,
    ISBN13: 978-0071232654,
    LC: QA76.73.C15.Q55.

  Local parameters:

    Local, double DIFF, the norm of the change in the solution from one iteration
    to the next.

    Local, double MEAN, the average of the boundary values, used to initialize
    the values of the solution in the interior.

    Local, double U[M][N], the solution at the previous iteration.

    Local, double W[M][N], the solution computed at the latest iteration.
*/
{
    thread_data_t data;
    double epsilon = 0.001;
    int i;
    int iterations;
    int iterations_print;
    int j;
    int num_threads = 4;
    enum schedule_type schedule = STATIC;
    int chunk_size = 64;

    if (argc > 1)
    {
        num_threads = atoi(argv[1]);
    }

    printf("\n");
    printf("HEATED_PLATE_Pthread\n");
    printf("  C/Pthread version\n");
    printf("  A program to solve for the steady state temperature distribution\n");
    printf("  over a rectangular plate.\n");
    printf("\n");
    printf("  Spatial grid of %d by %d points.\n", M, N);
    printf("  The iteration will be repeated until the change is <= %e\n", epsilon);
    printf("  Number of threads =              %d\n", num_threads);
    printf("  Schedule type =                 %s\n", schedule == STATIC ? "STATIC" : (schedule == DYNAMIC ? "DYNAMIC" : "GUIDED"));
    printf("  Chunk size =                   %d\n", chunk_size);
    printf("\n");

    data.mean = 0.0;
    data.my_diff = 0.0;
    data.diff = 0.0;
    pthread_mutex_init(&data.mutex, NULL);
    /*
      Set the boundary values, which don't change.
    */
    // w[0][j] = 0.0;
    parallel_for(0, N, 1, set_top_boundary, &data, num_threads, schedule, chunk_size);
    // w[M - 1][j] = 100.0;
    parallel_for(0, N, 1, set_bottom_boundary, &data, num_threads, schedule, chunk_size);
    // w[i][0] = 100.0;
    parallel_for(1, M - 1, 1, set_left_boundary, &data, num_threads, schedule, chunk_size);
    // w[i][N - 1] = 100.0;
    parallel_for(1, M - 1, 1, set_right_boundary, &data, num_threads, schedule, chunk_size);
    /*
      Average the boundary values, to come up with a reasonable
      initial value for the interior.
    */
    for (i = 1; i < M - 1; i++)
    {
        data.mean = data.mean + data.w[i][0] + data.w[i][N - 1];
    }
    for (j = 0; j < N; j++)
    {
        data.mean = data.mean + data.w[M - 1][j] + data.w[0][j];
    }

    /*
      Pthread note:
      You cannot normalize MEAN inside the parallel region.  It
      only gets its correct value once you leave the parallel region.
      So we interrupt the parallel region, set MEAN, and go back in.
    */
    data.mean = data.mean / (double)(2 * M + 2 * N - 4);
    printf("\n");
    printf("  MEAN = %f\n", data.mean);
    /*
      Initialize the interior solution to the mean value.
    */
    // w[i][j] = mean;
    parallel_for(1, M - 1, 1, set_initial_mean, &data, num_threads, schedule, chunk_size);
    /*
      Set the initial value of the solution U to the mean value.
    */

    /*
      iterate until the  new solution W differs from the old solution U
      by no more than EPSILON.
    */
    iterations = 0;
    iterations_print = 1;
    printf("\n");
    printf(" Iteration  Change\n");
    printf("\n");
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    data.diff = epsilon;

    while (epsilon <= data.diff)
    {
        parallel_for(0, M, 1, copy_solution, &data, num_threads, schedule, chunk_size);
        /*
          Save the old solution in U.
        */

        /*
          Determine the new estimate of the solution at the interior points.
          The new solution W is the average of north, south, east and west neighbors.
        */

        parallel_for(1, M - 1, 1, compute_new_solution, &data, num_threads, schedule, chunk_size);
        /*
          C and C++ cannot compute a maximum as a reduction operation.

          Therefore, we define a private variable MY_DIFF for each thread.
          Once they have all computed their values, we use a CRITICAL section
          to update DIFF.
        */
        data.diff = 0.0;
        parallel_for(1, M - 1, 1, compute_diff, &data, num_threads, schedule, chunk_size);
        iterations++;
        if (iterations == iterations_print)
        {
            printf("  %8d  %f\n", iterations, data.diff);
            iterations_print = 2 * iterations_print;
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double wtime = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    printf("\n");
    printf("  %8d  %f\n", iterations, data.diff);
    printf("\n");
    printf("  Error tolerance achieved.\n");
    printf("  Wallclock time = %f\n", wtime);
    /*
      Terminate.
    */
    printf("\n");
    printf("HEATED_PLATE_Pthread:\n");
    printf("  Normal end of execution.\n");

    return 0;
}
