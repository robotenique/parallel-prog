NVCC = nvcc
CC = g++
CFLAGS = -lm -O2 -w
LDFLAGS = ""
.PHONY: clean

all: main

clean:
	rm -f *.o *~ *.out teste main

main: reduce.o error.o macros.o
	$(NVCC) -o $@ $^ $(CLFLAGS)

teste: teste.o error.o cudaerror.o
	$(NVCC) -o $@ $^ $(CLFLAGS)

%.o: %.cu %.h
	$(NVCC) $(CFLAGS) -c $< $(CLFLAGS)

%.o: %.cu
	$(NVCC) $(CFLAGS) -c $< $(CLFLAGS)

%.o: %.cpp %.h
	$(CC) $(CFLAGS) -c $< $(CLFLAGS) -o $@

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< $(CLFLAGS) -o $@
#DEVICE_ID = 0
#sizeV=262144
#sizeM=2048
