NVCC = nvcc
CC = nvcc
CFLAGS = -w -Xptxas --opt-level=3 -arch sm_30
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
