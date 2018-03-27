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
int N, var, freeRock;
volatile bool foundDeadlock;
volatile int counter;

typedef struct frog_t {
    int pos, id;
} Frog;


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

    while (!foundDeadlock && counter < var*N) {
        next = pos + type;
        lastPos = pos;
        if (next < N && next >= 0 && !board[next]) {
            P(board_mtx + next);
            if (!board[next]) {
                P(board_mtx + pos);
                oldPos = pos;
                board[pos] = 0;
                freeRock = pos;
                board[next] = id;
                pos = next;
                V(board_mtx + oldPos);
                P(&counter_mtx);
                counter = 0;
                V(&counter_mtx);
            }
            V(board_mtx + next);
        }
        if (lastPos == pos && (next += type) < N && next >= 0 && !board[next]) {
            P(board_mtx + next);
            if (!board[next]) {
                P(board_mtx + pos);
                oldPos = pos;
                board[pos] = 0;
                freeRock = pos;
                board[next] = id;
                pos = next;
                V(board_mtx + oldPos);
                P(&counter_mtx);
                counter = 0;
                V(&counter_mtx);

            }
            V(board_mtx + next);
        }
        if (lastPos == pos) {
            P(&counter_mtx);
            counter++;
            V(&counter_mtx);
        }
    }
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

void cleanArray(bool *arr){
    for (int i = 0; i < N - 1; i++)
        arr[i] = false;
}

void freeAll(){

}

int main(int argc, char const *argv[]) {
    set_prog_name("frog-puzzle");
    srand(time(NULL));

    var = 1000;
    bool ok = true;
    int noTests = 10;
    int success = 0;
    // number of stones in total
    //N = 6;
    if(argc < 2)
        die("Wrong number of arguments!\nUsage ./puzzle <number of rocks>");
    N = atoi(argv[1]);
    if (N%2==0)
        die("Number of rocks must be an odd integer...");

    // frogs
    int mFrogs = (N - 1)/2;
    int fFrogs = (N - 1)/2;
    Frog *frogs;
    freeRock = N/2;

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
        }
        board[mFrogs] = 0;
        foundDeadlock = false;

        pthread_barrier_wait(&bar);
        // The main thread is the manager thread to check deadlock
        while (counter < var*N && !foundDeadlock) {
            P(&counter_mtx);
                if(counter >= var*N)
                    foundDeadlock = true;
            V(&counter_mtx);
            // Check if a frog can move to the freeRock
            if(board[freeRock] == 0) {
                if(freeRock > 0 && board[freeRock - 1] > 0)
                    continue;
                else if(freeRock > 1 && board[freeRock - 2] > 0)
                    continue;
                else if(freeRock < N - 1 && board[freeRock + 1] < 0)
                    continue;
                else if(freeRock < N - 2 && board[freeRock + 2] < 0)
                    continue;
                else
                    foundDeadlock = true;
            }
        }
        pthread_barrier_wait(&bar);

        // Verify if the game was solved or not
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
    free(frogs);
    free(board_mtx);
    pthread_exit(NULL);
}
