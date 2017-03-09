target: all

all:
	cd seq && make
	cd mpi && make
	cd omp && make
	cd mpi_omp && make

.PHONY: clean
clean:
	cd seq && make clean
	cd mpi && make clean
	cd omp && make clean
	cd mpi_omp && make clean
