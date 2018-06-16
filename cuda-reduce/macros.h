#ifndef __MACROS_H__
#define __MACROS_H__

#include <cinttypes>
#include <string>

#define NUM_THREADS 256

/*
 * Function: new_matrix_from_file
 * --------------------------------------------------------
 * Read a file containing matrices and store them at a vector
 *
 * @args filename : The name of the file that the function will read
 *       dest : Pointer to where the function will store the matrices
 *
 * @return number of matrices
 */
int32_t new_matrix_from_file(std::string filename, int32_t** dest);

/*
 * Function: reduce_matrices_seq
 * --------------------------------------------------------
 * Reduce the matrices using a sequential algorithm
 *
 * @args list_m : Pointer to the matrices' list
 *       num_m : Number of matrices
 *       dest : Pointer to where the function will store the result
 *
 * @return
 */
void reduce_matrices_seq(int32_t *list_m, int32_t num_m, int32_t **dest);

/*
 * Function: print_matrix
 * --------------------------------------------------------
 * Print a matrix to the stdout
 *
 * @args mtx : Pointer to the matrix
 *
 * @return
 */
void print_matrix(int32_t *matx);

#endif
