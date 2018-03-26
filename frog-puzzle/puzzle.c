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
#include <time.h>
#include "error.h"
#include "macros.h"

pthread_mutex_t *board_mtx, counter_mtx;
pthread_barrier_t bar;
pthread_t *threads;
int *board;
int counter, N, var;

typedef struct frog_t {
    int pos, id;
} Frog;

void debugPond(){
    printf("|");
    for (int i = 0; i < N - 1; i++)
        if (board[i] > 0)
            printf("\x1b[96m%2d\x1b[0m|", board[i]);
        else if (board[i] < 0)
            printf("\x1b[95m%2d\x1b[0m|", board[i]);
        else
            printf("%2d|", board[i]);
    if (board[N - 1] > 0)
        printf("\x1b[96m%2d\x1b[0m|\n", board[N - 1]);
    else if (board[N - 1] < 0)
        printf("\x1b[95m%2d\x1b[0m|\n", board[N - 1]);
    else
        printf("%2d|\n", board[N - 1]);


}

int sign(int x) {
    return (x < 0)? -1 : ((x == 0)? 0 : 1);
}

void *frog(void *args) {
    Frog *f = (Frog*)args;
    int pos = f->pos;
    int id = f->id;
    int type = sign(f->id);
    int next, lastPos, oldPos;

    // Wait for all threads to be ready
    pthread_barrier_wait(&bar);

    while (counter < var*N) {
        next = pos + type;
        lastPos = pos;
        if (next < N && next >= 0 && !board[next]) {
            // printf("Frog %d is waiting to jump to %d\n", id, next);
            P(board_mtx + next);
            if (!board[next]) {
                // printf("%d: My destiny is free!\n", id);
                P(board_mtx + pos);
                oldPos = pos;
                // printf("Frog %d jumped from %d to %d\n", id, pos, next);
                board[pos] = 0;
                board[next] = id;
                pos = next;
                V(board_mtx + oldPos);
                // printf("ESPERANDO MUTEX counterNormal.... %d\n", id);
                P(&counter_mtx);
                counter = 0;
                V(&counter_mtx);
            }
            V(board_mtx + next);
        }
        if (lastPos == pos && (next += type) < N && next >= 0 && !board[next]) {
            // printf("Frog %d is waiting to jump to %d\n", id, next);
            P(board_mtx + next);
            if (!board[next]) {
                // printf("%d: My destiny is free!\n", id);
                P(board_mtx + pos);
                oldPos = pos;
                // printf("Frog %d jumped from %d to %d\n", id, pos, next);
                board[pos] = 0;
                board[next] = id;
                pos = next;
                V(board_mtx + oldPos);
                // printf("ESPERANDO MUTEX (counter).... %d\n", id);
                P(&counter_mtx);
                counter = 0;
                V(&counter_mtx);
            }
            V(board_mtx + next);
        }
        if (lastPos == pos) {
            // printf("ESPERANDO MUTEX(counter lastPos).... %d\n", id);
            P(&counter_mtx);
            counter++;
            // printf("Frog %d failed...\n", id);
            V(&counter_mtx);
        }
        // pthread_barrier_wait(&bar);
        // pthread_barrier_wait(&bar);
    }
    // printf("Frog %d quited\n", id);
    //printf("ESPERANDO BARREIRA.... %d\n", id);
    pthread_barrier_wait(&bar);
    pthread_exit(NULL);
}

Frog createFrog(int pos, int id) {
    Frog f;
    f.pos = pos;
    f.id = id;
    return f;
}

Frog *createFrogs(int mFrogs, int fFrogs) {
    Frog *frogs;
    int n = mFrogs + fFrogs;
    frogs = emalloc(n*sizeof(Frog));
    for (int i = 0; i < mFrogs; i++)
        frogs[i] = createFrog(i, i + 1);
    for (int i = 0; i < fFrogs; i++)
        frogs[n - i - 1] = createFrog(n - i, -(i + 1));
    return frogs;
}

void shuffle(Frog *frogs, int n) {
     int j;
     Frog tmp;
     for (int i = n - 1; i > 0; i--) {
         j = rand() % (i + 1);
         tmp = frogs[j];
         frogs[j] = frogs[i];
         frogs[i] = tmp;
     }
}


int main(int argc, char const *argv[]) {
    set_prog_name("frog-puzzle");
    srand(time(NULL));

    var = 100;
    bool ok = true;
    int noTests = 100000;
    int success = 0;

    // number of stones in total
    N = 3;

    // frogs
    int mFrogs = (N - 1)/2;
    int fFrogs = (N - 1)/2;
    Frog *frogs;

    // initializing structs
    frogs = createFrogs(mFrogs, fFrogs);
    board = emalloc(N*sizeof(int));
    board_mtx = emalloc(N*sizeof(pthread_mutex_t));
    pthread_barrier_init(&bar, NULL, N);
    pthread_mutex_init(&counter_mtx, NULL);
    for (int i = 0; i < N; i++)
        pthread_mutex_init(board_mtx + i, NULL);

    for (int k = 0; k < noTests; k++) {
        counter = 0;
        threads = emalloc((N - 1)*sizeof(pthread_t));
        shuffle(frogs, mFrogs + fFrogs);
        for (int i = 0; i < mFrogs + fFrogs; i++) {
            pthread_create(threads + i, NULL, &frog, (void*)(frogs + i));
            pthread_detach(threads[i]);
            board[frogs[i].pos] = frogs[i].id;
            //printf("pos = %d, id = %d\n", frogs[i].pos, frogs[i].id);
        }
        board[mFrogs] = 0;

        //printf("I'M WAITING\n\n\n\n\n");
        pthread_barrier_wait(&bar);

        /*while (true) {
        pthread_barrier(&bar);
        // verifica && break
        pthread_barrier(&bar);
        }*/

        pthread_barrier_wait(&bar);

        //debugPond();

        // Verify if the game whether solved or not
        ok = true;
        for (int i = 0; i < fFrogs && ok; i++)
            ok &= (sign(board[i]) == -1);
        ok &= (sign(board[fFrogs]) == 0);
        for (int i = fFrogs + 1; i < N && ok; i++)
            ok &= (sign(board[i]) == 1);

        success += ok;

        free(threads);
    }

    printf("Success rate: %f%%\n", 100.0*success/noTests);

    free(board);
    free(board_mtx);
    pthread_exit(NULL);
}
