CC = nvcc
CFLAGS = -lm -O2 -w
LDFLAGS = ""
.PHONY: clean
all: teste

clean:
	rm -f *.o *~ *.out teste

teste: teste.o
	$(CC) -o $@ $^ $(LDFLAGS) $(CLFLAGS)

%.o: %.cu %.h
	$(CC) $(CFLAGS) -c $< $(LDFLAGS) $(CLFLAGS) -o $@


%.o: %.cu
	$(CC) $(CFLAGS) -c $< $(LDFLAGS) $(CLFLAGS) -o $@
#DEVICE_ID = 0
#sizeV=262144
#sizeM=2048
