#include <iostream>
#include <fstream>
#include <string>
#include "macros.h"
#include "error.h"
using namespace std;

int main(int argc, char const *argv[]) {
    set_prog_name("cuda-reduce");
    if(argc < 2)
        die("Wrong number of arguments!\nUsage ./main <path_matrices_file>");
    int32_t *list_m;
    int32_t num_m = new_matrix_from_file(argv[1], list_m);
    cout << "OLAR";
    cout << num_m;
    print_matrices(list_m, num_m);
    return 0;
}
