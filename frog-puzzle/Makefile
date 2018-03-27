#
# @author: Juliano Garcia de Oliveira Nº USP: 9277086
# @author: João Gabriel Basi Nº USP: 9793801
#
CC=gcc
CFLAGS = -Wall -pedantic -O2 -std=gnu11
LDFLAGS = -lrt -lm -pthread

.PHONY: clean


all: puzzle


puzzle: puzzle.o error.o macros.o
	$(CC) -o $@ $^ $(LDFLAGS)

%.o: %.c %.h
	$(CC) $(CFLAGS) -c $< $(LDFLAGS) -o $@


%.o: %.c
	$(CC) $(CFLAGS) -c $< $(LDFLAGS) -o $@


clean:
	rm -f *.o *~ *.out puzzle
