/*
 * @author: João Gabriel Basi Nº USP: 9793801
 * @author: Juliano Garcia de Oliveira Nº USP: 9277086
 *
 * MAC0219
 * 26/03/19
 *
 * A header with some macros
 */

#ifndef __MACROS_H__
#define __MACROS_H__

#include <pthread.h>
#include <math.h>

#define NANO_CONVERT 1e-9

#define P(x) pthread_mutex_lock(x)
#define V(x) pthread_mutex_unlock(x)

/* Simple types definition */
typedef enum { false, true } bool;

typedef unsigned int u_int;

typedef unsigned long long int u_lint;

/*
 * Function: sleepFor
 * --------------------------------------------------------
 * Sleeps for dt seconds
 *
 * @args dt : seconds to sleep
 *
 * @return
 */
void sleepFor(double);


#endif
