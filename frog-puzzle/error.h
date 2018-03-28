/*
 * @author: João Gabriel Basi Nº USP: 9793801
 * @author: Juliano Garcia de Oliveira Nº USP: 9277086
 *
 * MAC0219
 * 26/03/19
 *
 * Error-handling routines.
 * This code is an adaptation of the module provided during
 * our MAC216 course.
 * https://www.ime.usp.br/~fmario/cursos/mac216-15/
 */

#ifndef __ERROR_H__
#define __ERROR_H__

#include <stdlib.h>
#include <stdio.h>

/*
 * Function: set_prog_name
 * --------------------------------------------------------
 * Set the program name.
 *
 * @args name : string
 *
 * @return void
 */
void set_prog_name(const char *name);

/*
 * Function: get_error_msg
 * --------------------------------------------------------
 * Return the error message previously set.
 *
 * @return error message
 */
const char *get_error_msg();

/*
 * Function: set_error_msg
 * --------------------------------------------------------
 * Set error message to msg, formatted with arguments as in sprintf.
 *
 * @args msg : const char*
 *
 * @return void
 */
void set_error_msg(const char *msg, ...);

/*
 * Function: print_error_msg
 * --------------------------------------------------------
 * Print error message in stderr.
 *
 * If pointer msg is NULL, then the error message that has been set is
 * used; otherwise the message given is used, formatted with arguments
 * as in sprintf.
 *
 * If the message ends in a colon (:) then the system's error message is
 * concatenated, in case errno is nonzero; otherwise the colon is
 * removed and replaced by a period.
 *
 * @args msg : const char*
 *
 * @return void
 */
void print_error_msg(const char *msg, ...);

/*
 * Function: die
 * --------------------------------------------------------
 * Print error message like print_error_msg and exit the program.
 *
 * @args msg : const char*
 *
 * @return void
 */
void die(const char *msg, ...);

/*
 * Function: emalloc
 * --------------------------------------------------------
 * Like malloc, but exit the program with an error message on
 * failure.
 *
 * @args size : size_t
 *
 * @return a void pointer to the allocated block
 */
void *emalloc(size_t size);

/*
 * Function: efopen
 * --------------------------------------------------------
 * Like fopen, but exit the program with an error message on
 * failure.
 *
 * @args filename : the name of the file
 *       mode : the mode to open the file
 *
 * @return a pointer to the opened file
 */
FILE *efopen(const char *filename, const char *mode);

/*
 * Function: estrdup
 * --------------------------------------------------------
 * Like strdup, but crashes the program with an error message on
 * failure.
 *
 * @args s : const char*
 *
 * @return a pointer to the copy of the string
 */
char *estrdup(const char *s);

#endif
