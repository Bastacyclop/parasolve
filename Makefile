target: decide

CFLAGS= -g -O3 -std=c99 -Wall

aux.o: aux.c projet.h

parasolve_mpi%: parasolve_mpi%.c aux.o
	mpicc $(CFLAGS) $^ -o $@

main.o: main.c projet.h
decide: main.o aux.o
	$(CC) $(CFLAGS) $^ -o $@

.PHONY: clean
clean:
	rm -f *.o decide

