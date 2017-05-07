# Parasolve

## Running things

```sh
# compiling everything (optional)
make
# cleaning up compilation artifacts
make clean
# there is a dedicated Makefile inside each implementation directories

# and also some handy scripts
# run a decision implementation
[OMP_NUM_THREADS=omp_n] ./run.py [implementation_directory [input [mpi_n]]]
```
