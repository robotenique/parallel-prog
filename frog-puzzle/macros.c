#include "macros.h"

void sleepFor(double dt){
    nanosleep(&(struct timespec){floor(dt),(long)((dt-floor(dt))/NANO_CONVERT)}, NULL);
}
