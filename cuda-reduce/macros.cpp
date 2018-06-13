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
    streambuf *cinbuf = cin.rdbuf(in.rdbuf()); // Redirects cin to the file

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
    }

    cin.rdbuf(cinbuf); // Redirects back cin to stdin

    return n;
}

int32_t minT(int32_t a, int32_t b) {
    return b + ((a-b)&((a-b) >> 31));
}
