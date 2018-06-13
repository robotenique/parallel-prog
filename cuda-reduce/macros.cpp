#include <cstdlib>
#include <cinttypes>
#include <fstream>
#include <string>
#include <iostream>
#include "macros.h"
#include "error.h"

using namespace std;

int32_t new_matrix_from_file(string filename, int32_t* dest) {
    ifstream in(filename);
    streambuf *cinbuf = cin.rdbuf(in.rdbuf()); // set cin buffer to 'filename' content
    string dummy;
    int32_t n;

    cin >> n;
    cin >> dummy;
    dest = (int32_t*)emalloc(9*n*sizeof(int32_t));

    for (int i = 0; i < 9*n; i += 9) {
        cin >> dest[i] >> dest[i+1] >> dest[i+2];
        cin >> dest[i+3] >> dest[i+4] >> dest[i+5];
        cin >> dest[i+6] >> dest[i+7] >> dest[i+8];
        cin >> dummy;
        cout << "OLAR";
    }

    cin.rdbuf(cinbuf); // set cin buffer back to stdin

    return n;
}

int32_t minT(int32_t a, int32_t b) {
    return b + ((a-b)&((a-b) >> 31));
}

void print_matrices(int32_t *list_m, int32_t num_m) {
    cout << "I = "  << " ";
    for (size_t i = 0; i < 9*num_m; i++) {

        cout << list_m[i] << " ";
        if ((i + 1)%9 == 0)
            cout << endl;
    }
}
