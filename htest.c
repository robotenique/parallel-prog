#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>
#include <float.h>
#include <time.h>
#define MAX_ITER 90000000
#define NUM_THREADS 5

pthread_barrier_t bar;
pthread_t t1, t2, t3, t4, t5;
pthread_mutex_t mut1;
volatile unsigned long int counter;
volatile float var1, var2, var3, var4, globalAcc;

void *calcDummy(void *args) {
	int id = *(int*)args;
	pthread_barrier_wait(&bar);
	float myX = 0.5;
	float myY = 10.2;
	float acc = 0;
	int myCounter = 0;
	while(myCounter < MAX_ITER/NUM_THREADS){
		if(rand()%2) {
			acc += exp(acc)*sqrtf(tanh(sinh(cosh(sin(var1*var2*var3*exp(var3))))));
			myX *= 0.1*acc;
			myY *= 1/(acc+1) + 1/exp(myX);
		}
		else {
			acc -= powf(exp(var1)*tanh(var4), exp(sin(tanh(exp(acc)))));
			myX /= (acc+1);
			myY /= acc + exp(myX);
		}
		if(acc >= 0.7*FLT_MAX)
			acc = 0.4*acc;
		else if(acc <= 0.7*FLT_MIN)
			acc = fabs(acc);
		if(myX >= 0.7*FLT_MAX)
			myX = 0.4*myX;
		else if(myX <= 0.7*FLT_MIN)
			myX = fabs(myX);
		if(myY >= 0.7*FLT_MAX)
			myY = 0.4*myY;
		else if(myY <= 0.7*FLT_MIN)
			myY = fabs(myY);

		if(isinf(acc))
			acc = 0.1898127*(rand()%MAX_ITER);

		if(myCounter%1000 == 0) {
			pthread_mutex_lock(&mut1);
				globalAcc += 0.11237189*acc;
				globalAcc = 0.1*globalAcc;
			pthread_mutex_unlock(&mut1);
		}
		myCounter++;
	}
	pthread_barrier_wait(&bar);
	pthread_exit(NULL);
}



int main() {
	srand(time(NULL));
	// Threads ID
	int id1 = 1;
	int id2 = 2;
	int id3 = 3;
	int id4 = 4;
	int id5 = 5;
	counter = 0; // Maybe this is useless...
	// Parameters
	var1 = 1.2;
	var2 = 90.1;
	var3 = 20.2;
	var4 = 12.3;


	globalAcc = 0;
	clock_t t;
	t = clock();
	pthread_mutex_init(&mut1, NULL);
	pthread_barrier_init(&bar, NULL, NUM_THREADS + 1);
	pthread_create(&t1, NULL, &calcDummy, (void*)(&id1));
	pthread_create(&t2, NULL, &calcDummy, (void*)(&id2));
	pthread_create(&t3, NULL, &calcDummy, (void*)(&id3));
	pthread_create(&t4, NULL, &calcDummy, (void*)(&id4));
	pthread_create(&t5, NULL, &calcDummy, (void*)(&id5));
	pthread_detach(t1);
	pthread_detach(t2);

    // Wait for all threads to be ready
    pthread_barrier_wait(&bar);

	// Wait for all threads to finish
	pthread_barrier_wait(&bar);
	t = clock() - t;
	double time_taken = 100*((double)t)/CLOCKS_PER_SEC;
	printf("\ntime of calculation = %fms\n\n", time_taken);

	return 0;
}
