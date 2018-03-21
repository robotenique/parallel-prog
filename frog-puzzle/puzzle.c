/*
 * @author: João Gabriel Basi Nº USP: 9793801
 * @author: Juliano Garcia de Oliveira Nº USP: 9277086
 *
 * MAC0219
 * 26/03/19
 *
 *
 */
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include "error.h"
#include "macros.h"


int main(int argc, char const *argv[]) {
    set_prog_name("frog-puzzle");
    // number of stones in total
    int N = 5;
    // frogs
    int mFrogs = (N - 1)/2;
    int fFrogs = (N - 1)/2;
    pthread_mutex_t* board = emalloc(N*sizeof(pthread_mutex_t));
    for (int i = 0; i < N; i++)
        pthread_mutex_init(&(board[i]), NULL);
    





    return 0;
}
