#include <stdio.h>
#include <time.h>
#include <omp.h>
#include <stdlib.h>

#ifdef __CYGWIN__
# define CLOCK CLOCK_MONOTONIC
#else
# define CLOCK CLOCK_MONOTONIC_RAW
#endif

#define THREAD_COUNT 8

/**
 * Converts the time spec to milliseconds.
 *
 * @param ts - A pointer to a timespec
 * @return The wall time in milliseconds
 */
double ts_to_ms(struct timespec* ts) {
    return (((double) ts->tv_sec) * 1000.0) + (((double) ts->tv_nsec) / 1000000.0);
}

/**
 * Initializes the passed in array with doubles.
 * The array is of size n. Used to initialize the
 * x and y vectors.
 * @param arr - the array (vector) to initialize
 * @param n - the size of the array (vector)
 * @param threads - the number of threads for the team
 */
void initialize_array(double* arr, int n, int threads) {
#    pragma omp parallel for num_threads(threads)
    for (int i = 0; i < n; i++) {
        arr[i] = rand(); // will initialize all to the same value
    }
}

/**
 * Returns a random integer value, which is stored
 * in a double. The size of the double is what we
 * are concerned with in this example. The actual numbers
 * do not matter.
 * @return a random scalar
 */
double initialize_scalar() {
    return rand();
}

/**
 * Multiplies the x vector by the scalar value a. Then,
 * sums the dilated vector with the y vector. The final
 * result is mapped back into the y vector.
 *
 * @param x - the first vector, which is multiplied by a scalar
 * @param y - the second vector, and the storage vector
 * @param a - the scalar to multiply x by
 * @param n - the size of the x and y vectors
 * @param threads - the number of threads for the team
 */
void calculate_daxpy(double* x, double*y, double a, int n, int threads) {
#    pragma omp parallel for num_threads(threads) schedule(static)
    for (int i = 0; i < n; i++) {
        y[i] = y[i] + x[i]*a;
    }
}

/**
 * Multiplies the x vector by the scalar value a. Then,
 * sums the dilated vector with the y vector. The final
 * result is mapped back into the y vector.
 *
 * @param x - the first vector, which is multiplied by a scalar
 * @param y - the second vector, and the storage vector
 * @param a - the scalar to multiply x by
 * @param n - the size of the x and y vectors
 * @param threads - the number of threads for the team
 */
void calculate_daxpy_dynamic(double* x, double*y, double a, int n, int threads) {
#    pragma omp parallel for num_threads(threads) schedule(dynamic, 100)
    for (int i = 0; i < n; i++) {
        y[i] = y[i] + x[i]*a;
    }
}

/**
 * Prints the daxby result to the screen.
 * The for loop is commented out since it was used
 * primarily for testing. The elements don't matter.
 *
 * @param y - the calculated daxby result vector
 * @param n - the size of the vector
 * @param proc_time - the processor time
 * @param wall_time - the wall clock time (human time)
 */
void print_daxpy(double* y, int n, double proc_time, double wall_time) {
//#    pragma omp single
//    for (int i = 0; i < n; i++) {
//        printf("The value of the y[%d] element: %f\n", i, y[i]);
//    }
    printf("Processor Time: %f\nWall Time: %f\n\n", proc_time, wall_time);
}

/**
 * The main method is the entry point of the program.
 * It seeds the random number generator before calling
 * the functions to initialize the vectors and calculate
 * the daxpy. Furthermore, it calculates the performance.
 *
 * The most significant portion of the work is calculating
 * the daxby result. This is where the bulk of the experimentation
 * will be performed.
 *
 * @return The exit code, indicating success or failure
 */
int main()
{
    // seed the random number generator.
    srand(time(NULL));

    struct timespec start, finish;

    // declare loop control variable
    int n = 1, thread_count = 1, dyn_or_stc;
    // read in the number n, which is the vector size
    while (n != -1) {
        printf("Run in static or dynamic mode? 0 is static: \n");
        scanf("%d", &dyn_or_stc);
        printf("Please input the number of threads: \n");
        scanf("%d", &thread_count);
        printf("Please input the size of the y and x vectors: \n");
        scanf("%d", &n);
        if (n != -1) {
            double a = initialize_scalar();
            // use malloc to support very large vector sizes
            // via the heap
            double* x = (double*) malloc(sizeof(double) * n);
            double* y = (double*) malloc(sizeof(double) * n);

            initialize_array(x, n, thread_count);
            initialize_array(y, n, thread_count);

            // Time the main calculation only
            clock_t begin = clock();
            clock_gettime(CLOCK, &start);
            if (dyn_or_stc == 0) {
                calculate_daxpy(x, y, a, n, thread_count);
            } else {
                calculate_daxpy_dynamic(x, y, a, n, thread_count);
            }
            clock_t end = clock();
            clock_gettime(CLOCK, &finish);
            // End of main static calculation

            // calculate times: time_spent is processor. wall_time is wall clock time
            double time_spent = (double) (end - begin) / CLOCKS_PER_SEC;
            double start_time = ts_to_ms(&start);
            double finish_time = ts_to_ms(&finish);
            double wall_time = finish_time - start_time;
            print_daxpy(y, n, time_spent, wall_time);
            free(x);
            free(y);
        }
    }


    // exit the program
    printf("Goodbye!\n");
    return 0;
}