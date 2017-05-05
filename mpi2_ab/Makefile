target: decide

CFLAGS= -g -O3 -std=c99 -Wall -lm

aux.o: aux.c projet.h

decide: main.c aux.o
	mpicc $(CFLAGS) $^ -o $@

exec: decide
	mpirun -n 4 ./decide "/ppp//PPP//7k//7K w"

.PHONY: clean
clean:
	rm -f *.o decide
