target: decide

CFLAGS= -g -O3 -std=c99 -Wall

aux.o: aux.c projet.h

parasolve_mpi: parasolve_mpi.c aux.o
	mpicc $(CFLAGS) $^ -o $@

parasolve_omp: parasolve_omp.c aux.o
	$(CC) $(CFLAGS) -fopenmp $^ -o $@

main.o: main.c projet.h
decide: main.o aux.o
	$(CC) $(CFLAGS) $^ -o $@

.PHONY: clean
clean:
	rm -f *.o decide parasolve_mpi parasolve_omp

