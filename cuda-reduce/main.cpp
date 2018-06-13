#include <iostream>    *dest = new int32_t[9*n];

#include <fstream>
#include <string>
#include "macros.h"
#include "error.h"
using namespace std;


int main(int argc, char const *argv[]) {
    set_prog_name("cuda-reduce");
    if(argc < 2)
        die("Wrong number of arguments!\nUsage ./main <path_matrices_file>");
    int32_t *list_m, *mat_reduced;
    int32_t num_m = new_matrix_from_file(argv[1], &list_m);
    print_matrices(list_m, num_m);
    reduce_matrices_seq(list_m, num_m, &mat_reduced);
    std::cout << "=======REDUCED=======" << '\n';
    print_matrices(mat_reduced, 1);
    delete[] list_m;
    delete[] mat_reduced;
    return 0;
}
