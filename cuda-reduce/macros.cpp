#include <cstdlib>
#include <cinttypes>
#include <fstream>
#include <string>
#include <iostream>
#include <limits>
#include "macros.h"
#include "error.h"

using namespace std;

int32_t new_matrix_from_file(string filename, int32_t** dest) {
    ifstream in(filename);
    streambuf *cinbuf = cin.rdbuf(in.rdbuf()); // set cin buffer to 'filename' content
    string dummy;
    int32_t n, round_n, i;

    cin >> n;
    cin >> dummy;
    round_n = n + NUM_THREADS - n%NUM_THREADS;
    *dest = new int32_t[9*round_n];
    int32_t* arr = *dest;

    for (i = 0; i < 9*n; i += 9) {
        cin >> arr[i] >> arr[i+1] >> arr[i+2];
        cin >> arr[i+3] >> arr[i+4] >> arr[i+5];
        cin >> arr[i+6] >> arr[i+7] >> arr[i+8];
        cin >> dummy;
    }

    for ( ; i < 9*round_n; i++)
        arr[i] = numeric_limits<int32_t>::max();

    cin.rdbuf(cinbuf); // set cin buffer back to stdin

    return round_n;
}

__host__ __device__ int32_t minT(int32_t a, int32_t b) {
    return b + ((a-b)&((a-b) >> 31));
}

void reduce_matrices_seq(int32_t *list_m, int32_t num_m, int32_t **dest){
    *dest = new int32_t[9];
    int32_t *arr = *dest;
    for (size_t i = 0; i < 9; i++)
        arr[i] = list_m[i];
    if (num_m == 9)
        return;
    for (size_t i = 9; i < 9*num_m; i++)
        arr[i%9] = minT(arr[i%9], list_m[i]);
}


void print_matrices(int32_t *list_m, int32_t num_m) {
    for (size_t i = 0; i < 9*num_m; i++) {
        cout << list_m[i] << " ";
        if ((i + 1)%9 == 0)
            cout << endl;
    }
}
