#
# @author: Juliano Garcia de Oliveira Nº USP: 9277086
# @author: João Gabriel Basi Nº USP: 9793801
#
CC=gcc
CFLAGS = -Wall -pedantic -O4 -std=gnu11
LDFLAGS = -lrt -lm -fopenmp -pthread -funroll-loops

.PHONY: clean


all: main


main: matmul.o error.o macros.o openmpmatmul.o ptmatmul.o
	$(CC) -o $@ $^ $(LDFLAGS) $(CLFLAGS)

%.o: %.c %.h
	$(CC) $(CFLAGS) -c $< $(LDFLAGS) $(CLFLAGS) -o $@


%.o: %.c
	$(CC) $(CFLAGS) -c $< $(LDFLAGS) $(CLFLAGS) -o $@


clean:
	rm -f *.o *~ *.out main
