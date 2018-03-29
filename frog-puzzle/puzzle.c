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

// Print the stone vector with frogs IDs
void debugPond(){
    printf("|");
    for (int i = 0; i < N; i++)
        if (board[i] > 0)
            printf("\x1b[96m%2d\x1b[0m|%s", board[i], (i == N - 1)? "\n" : "");
        else if (board[i] < 0)
            printf("\x1b[95m%2d\x1b[0m|%s", board[i], (i == N - 1)? "\n" : "");
        else
            printf("%2d|%s", board[i], (i == N - 1)? "\n" : "");
}

// Get the sign of a number
int sign(int x) {
    return (x < 0)? -1 : ((x == 0)? 0 : 1);
}

// Frog thread function
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
        // Check if an adjacent position exists and is available
        if (next < N && next >= 0 && !board[next]) {
            P(board_mtx + next);
            if (!board[next]) {
                // Change this frog position and change the value of some
                // control variables
                P(board_mtx + pos);
                oldPos = pos;
                board[pos] = 0;
                freeRock = pos;
                board[next] = id;
                pos = next;
                V(board_mtx + oldPos);
                // Reset the counter variable
                P(&counter_mtx);
                counter = 0;
                V(&counter_mtx);
            }
            V(board_mtx + next);
        }
        // Check if this frog can jump another one to land at an empty rock
        if (lastPos == pos && (next += type) < N && next >= 0 && !board[next]) {
            P(board_mtx + next);
            if (!board[next]) {
                // Change this frog position and change the value of some
                // control variables
                P(board_mtx + pos);
                oldPos = pos;
                board[pos] = 0;
                freeRock = pos;
                board[next] = id;
                pos = next;
                V(board_mtx + oldPos);
                // Reset the counter variable
                P(&counter_mtx);
                counter = 0;
                V(&counter_mtx);
            }
            V(board_mtx + next);
        }
        // If this couldn't move, add one to the counter variable
        if (lastPos == pos) {
            P(&counter_mtx);
            counter++;
            V(&counter_mtx);
        }
    }

    // Wait for all threads to end, so main can continue
    pthread_barrier_wait(&bar);
    pthread_exit(NULL);
}

// Create a single frog at position "pos" and ID "id"
Frog createFrog(int pos, int id) {
    Frog f;
    f.pos = pos;
    f.id = id;
    return f;
}

// Create a vector with "mFrogs" male frogs and "fFrogs" female frogs
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

// Shuffle a vector of frogs with size n
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

    Frog *frogs;
    int noTests = 10000;
    int success = 0;
    bool ok = true;
    var = 10000;

    if (argc < 2)
        die("Wrong number of arguments!\nUsage ./puzzle <number of rocks>");
    N = atoi(argv[1]);
    if (N%2 == 0)
        die("Number of rocks must be an odd integer...");

    int mFrogs = (N - 1)/2;
    int fFrogs = (N - 1)/2;
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

        // Shuffle frogs' vector, so the threads are initialized randomly
        shuffle(frogs, mFrogs + fFrogs);

        for (int i = 0; i < mFrogs + fFrogs; i++) {
            pthread_create(threads + i, NULL, &frog, (void*)(frogs + i));
            pthread_detach(threads[i]);
            board[frogs[i].pos] = frogs[i].id;
        }
        board[mFrogs] = 0;
        foundDeadlock = false;

        // Wait for all threads to be ready
        pthread_barrier_wait(&bar);

        // The main thread is the manager thread to check deadlock
        while (counter < var*N && !foundDeadlock) {
            // Check if a frog can move to the freeRock
            if (board[freeRock] == 0) {
                if (freeRock > 0 && board[freeRock - 1] > 0)
                    continue;
                else if (freeRock > 1 && board[freeRock - 2] > 0)
                    continue;
                else if (freeRock < N - 1 && board[freeRock + 1] < 0)
                    continue;
                else if (freeRock < N - 2 && board[freeRock + 2] < 0)
                    continue;
                else
                    foundDeadlock = true;
            }
        }

        // Wait for other threads to end
        pthread_barrier_wait(&bar);

        // Print pond
        debugPond();

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
