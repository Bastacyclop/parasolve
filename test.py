#!/bin/python

import os

input = '"4k//4K//4P w"'
if (os.environ.get('MPI') == "1"):
    os.system("make parasolve_mpi && time mpirun -n 4 parasolve_mpi {}".format(input))
if (os.environ.get('MPI') == "2"):
    os.system("make parasolve_mpi_bis && time mpirun -n 4 parasolve_mpi_bis {}".format(input))
else:
    os.system("make decide && time ./decide {}".format(input))
