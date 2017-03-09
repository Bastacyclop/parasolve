#!/bin/python

import os

input = os.environ.get('INPUT')
if input == None:
    input = '"4k//4K//4P w"'

cmd = "cd seq && make && ./decide {}"
if (os.environ.get('MPI') == "1"):
    cmd = "cd mpi && make && mpirun -n 5 decide {}"
elif (os.environ.get('OMP') == "1"):
    cmd = "cd omp && make && ./decide {}"
elif (os.environ.get('MPI_OMP') == "1"):
    cmd = "cd mpi_omp && make && mpirun -n 5 decide {}"

os.system(cmd.format(input))
