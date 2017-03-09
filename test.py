#!/bin/python

import os

input = '"4k//4K//4P w"'
if (os.environ.get('MPI') == "1"):
    os.system("make parasolve_mpi && mpirun -n 4 parasolve_mpi {}".format(input))
elif (os.environ.get('MPI') == "2"):
    os.system("make parasolve_mpi_bis && mpirun -n 4 parasolve_mpi_bis {}".format(input))
elif (os.environ.get('OMP') == "1"):
    os.system("make parasolve_omp && ./parasolve_omp {}".format(input))
else:
    os.system("make decide && ./decide {}".format(input))
