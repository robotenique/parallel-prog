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

#define VAR 5

pthread_mutex_t *board_mtx, counter_mtx;
pthread_barrier_t bar;
pthread_t *frogs;
int *board;
int counter, N;

struct frog_t {
    int pos, id;
};

int sign(int x) {
    return (x < 0)? -1 : 1;
}

void *frog(void *args) {
    struct frog_t *f = (struct frog_t*)args;
    int pos = f->pos;
    int id = f->id;
    int type = sign(f->id);
    int next, lastPos;
    pthread_barrier_wait(&bar);
    while (counter < VAR*N) {
        next = pos + type;
        lastPos = pos;
        if (next < N && next >= 0) {
            P(board_mtx + next);
            if (!board[next]) {
                P(board_mtx + pos);
                printf("Frog %d jumped from %d to %d\n", id, pos, next);
                board[pos] = 0;
                board[next] = id;
                pos = next;
                V(board_mtx + pos);
                P(&counter_mtx);
                counter = 0;
                V(&counter_mtx);
            }
            else {
                P(&counter_mtx);
                counter++;
                V(&counter_mtx);
            }
            V(board_mtx + next);
        }
        if (lastPos == pos && (next += type) < N && next >= 0) {
            P(board_mtx + next);
            if (!board[next]) {
                P(board_mtx + pos);
                printf("Frog %d jumped from %d to %d\n", id, pos, next);
                board[pos] = 0;
                board[next] = id;
                pos = next;
                V(board_mtx + pos);
                P(&counter_mtx);
                counter = 0;
                V(&counter_mtx);
            }
            else {
                P(&counter_mtx);
                counter++;
                V(&counter_mtx);
            }
            V(board_mtx + next);
        }
    }
    free(args);
    pthread_barrier_wait(&bar);
    return NULL;
}

struct frog_t *create_frog(int pos, int id) {
    struct frog_t *f = emalloc(sizeof(struct frog_t));
    f->pos = pos;
    f->id = id;
    return f;
}


int main(int argc, char const *argv[]) {
    set_prog_name("frog-puzzle");
    // frogs
    N = 5;
    int mFrogs = (N - 1)/2;
    int fFrogs = (N - 1)/2;
    struct frog_t *f;
    // number of stones in total

    // initializing structs
    counter = 0;
    board = emalloc(N*sizeof(int));
    board_mtx = emalloc(N*sizeof(pthread_mutex_t));
    frogs = emalloc((N - 1)*sizeof(pthread_t));
    pthread_barrier_init(&bar, NULL, N);
    pthread_mutex_init(&counter_mtx, NULL);
    for (int i = 0; i < N; i++)
        pthread_mutex_init(board_mtx + i, NULL);
    for (int i = 0; i < mFrogs; i++) {
        f = create_frog(i, i + 1);
        pthread_create(frogs + i, NULL, &frog, (void*)f);
        board[i] = i + 1;
    }
    for (int i = 0; i < fFrogs; i++) {
        f = create_frog(N - i, -(i + 1));
        pthread_create(frogs + N - i, NULL, &frog, (void*)f);
        board[N - i] = -(i + 1);
    }
    board[mFrogs + 1] = 0;

    pthread_barrier_wait(&bar);
    pthread_barrier_wait(&bar);

    for (int i = 0; i < N; i++)
        printf("%d, ", board[i]);
    printf("\n");

    free(board);
    free(board_mtx);
    free(frogs);
    return 0;
}
