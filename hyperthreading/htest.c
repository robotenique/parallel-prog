#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include "error.h"

#define MAX_WRITING 900000
#define NUM_THREADS 5
#define BILLION 1000000000.0

pthread_barrier_t bar;
pthread_mutex_t mut1;

void *writeDummy(void *args) {
	int id = *(int*)args;
	int myCounter = 0;
	char file[] = "fileN.txt";
	const char *text = "Write this to the file";
	FILE *f;
	file[4] = id + '0';
	pthread_barrier_wait(&bar);
	while (myCounter < MAX_WRITING/NUM_THREADS){
		f = efopen(file, "w");
		fprintf(f, "Some text: %s\n", text);
		fclose(f);
		myCounter++;
	}
	pthread_barrier_wait(&bar);
	pthread_exit(NULL);
}


int main() {
	set_prog_name("hyper-threading-test");
	srand(time(NULL));

	struct timespec start, finish;
	double elapsed;

	pthread_t *threads = emalloc(NUM_THREADS*sizeof(pthread_t));

	pthread_mutex_init(&mut1, NULL);
	pthread_barrier_init(&bar, NULL, NUM_THREADS + 1);

	clock_gettime(CLOCK_MONOTONIC, &start);

	for (int i = 0; i < NUM_THREADS; i++) {
		pthread_create(threads + i, NULL, &writeDummy, (void*)(&i));
		pthread_detach(threads[i]);
	}

	// Wait for all threads to be ready
	pthread_barrier_wait(&bar);

	// Wait for all threads to finish
	pthread_barrier_wait(&bar);

	clock_gettime(CLOCK_MONOTONIC, &finish);

	elapsed = (finish.tv_sec - start.tv_sec);
	elapsed += (finish.tv_nsec - start.tv_nsec) / BILLION;
	printf("\nTime of execution = %fs\n\n", elapsed);

	free(threads);

	return 0;
}
