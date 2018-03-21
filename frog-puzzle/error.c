/*
 * @author: João Gabriel Basi Nº USP: 9793801
 * @author: Juliano Garcia de Oliveira Nº USP: 9277086
 *
 * MAC0219
 * 26/03/19
 *
 * Error functions library.
 */

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include "error.h"

static char prog_name[1024] = "";
static char error_msg[1024];
void set_prog_name(const char *name) {
    strncpy(prog_name, name, 1023);
}


const char *get_error_msg() {
    return error_msg;
}


void set_error_msg(const char *msg, ...) {
    va_list arglist;

    va_start(arglist, msg);
    vsnprintf(error_msg, 1024, msg, arglist);
    va_end(arglist);
}


void print_error_msg(const char *msg, ...) {
    if (msg) {
        va_list arglist;

        va_start(arglist, msg);
        vsnprintf(error_msg, 1024, msg, arglist);
        va_end(arglist);
    }

    int len = strlen(error_msg);

    if (error_msg[len - 1] == ':' && errno) {
        fprintf(stderr, "%s: %s %s\n", prog_name, error_msg, strerror(errno));
        return;
    }
    else if (error_msg[len - 1] == ':')
        error_msg[len - 1] = '.';

    fprintf(stderr, "%s: %s\n", prog_name, error_msg);
}


void die(const char *msg, ...) {
    if (msg) {
        va_list arglist;

        va_start(arglist, msg);
        vsnprintf(error_msg, 1024, msg, arglist);
        va_end(arglist);
    }

    int len = strlen(error_msg);

    if (error_msg[len - 1] == ':' && errno) {
        fprintf(stderr, "%s: %s %s\n", prog_name, error_msg, strerror(errno));
        exit(0);
    }
    else if (error_msg[len - 1] == ':')
    error_msg[len - 1] = '.';

    fprintf(stderr, "%s: %s\n", prog_name, error_msg);
    exit(0);
}


void *emalloc(size_t size) {
    errno = 0;
    void *ret = malloc(size);

    if (!ret) {
        print_error_msg("call to malloc failed:");
        exit(-1);
    }

    return ret;
}


char *estrdup(const char *s) {
    errno = 0;
    char *ret = emalloc((strlen(s) + 1)*sizeof(char));

    if (!ret) {
        print_error_msg("call to estrdup failed:");
        exit(-1);
    }

    strcpy(ret, s);
    ret[strlen(s)] = '\0';
    return ret;
}

FILE *efopen(const char *filename, const char *mode){
    FILE* ret = fopen (filename, mode);
    if (ret == NULL)
        die("Error opening file %s", filename);
    return ret;
}
