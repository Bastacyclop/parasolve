#!/bin/python

import os

input = os.environ.get('INPUT')
if input == None:
    input = '"4k//4K//4P w"'

cmd = "make decide && ./decide {}"
if (os.environ.get('MPI') == "1"):
    cmd = "make parasolve_mpi && mpirun -n 4 parasolve_mpi {}"
elif (os.environ.get('MPI') == "2"):
    cmd = "make parasolve_mpi_bis && mpirun -n 4 parasolve_mpi_bis {}"
elif (os.environ.get('OMP') == "1"):
    cmd = "make parasolve_omp && ./parasolve_omp {}"

os.system(cmd.format(input))
