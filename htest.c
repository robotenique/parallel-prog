#include <stdio.h>
#include <pthread.h>
#include <math.h>
pthread_barrier_t bar;
pthread_t t1, t2;
volatile int counter;
void *calcDummy(void *args) {
	int id = *(int*)args;
	printf("id = %d\n", id);
	pthread_barrier_wait(&bar);
}



int main() {
	printf("Olá\n");	
	int id1 = 1;
	int id2 = 2;
	pthread_barrier_init(&bar, NULL, 3);
	pthread_create(&t1, NULL, &calcDummy, (void*)(&id1));
    pthread_detach(&t1);
	pthread_create(&t2, NULL, &calcDummy, (void*)(&id2));
	pthread_detach(&t2);
    
    // Wait for all threads to be ready
    pthread_barrier_wait(&bar);

	
	return 0;
}