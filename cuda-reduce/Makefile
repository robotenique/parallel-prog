NVCC = nvcc
CC = g++
CFLAGS = -O2 -w
LDFLAGS = -lm
.PHONY: clean

all: main

clean:
	rm -f *.o *~ *.out main

main: main.o error.o cudaerror.o macros.o
	$(NVCC) -o $@ $^ $(LDFLAGS)

%.o: %.cu %.h
	$(NVCC) $(CFLAGS) $(LDFLAGS) -c $<

%.o: %.cu
	$(NVCC) $(CFLAGS) $(LDFLAGS) -c $<

%.o: %.cpp %.h
	$(CC) $(CFLAGS) $(LDFLAGS) -c $< -o $@

%.o: %.cpp
	$(CC) $(CFLAGS) $(LDFLAGS) -c $< -o $@

%.o: %.c
	$(CC) $(CFLAGS) $(LDFLAGS) -c $< -o $@
