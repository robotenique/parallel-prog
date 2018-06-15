#include <cstdio>
#include <iostream>
using namespace std;
int32_t min_coisa(int32_t a, int32_t b){
    return (a - abs(a-b) + b)/2;
}
int main(int argc, char const *argv[]) {
    cout << "MIN(-6, INT32_T-MAX) = " << min_coisa(-6, 2147483647) << endl;
    cout << "MIN_CPP(-6, INT32_T-MAX) = " << min(-6, 2147483647) << endl;
    return 0;
}
