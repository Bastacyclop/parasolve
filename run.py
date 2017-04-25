#!/bin/python

import os
import sys

input = '"4k//4K//4P w"'
if len(sys.argv) >= 3:
    input = sys.argv[2]

cmd = "cd seq && make && ./decide {}"
if len(sys.argv) >= 2:
    if (sys.argv[1] == 'mpi'):
        n = 4
        if len(sys.argv) >= 4:
            n = int(sys.argv[3])
        cmd = "cd mpi && make && mpirun -n {} decide {}".format(n, "{}")
    elif (sys.argv[1] == 'mpi2'):
        n = 4
        if len(sys.argv) >= 4:
            n = int(sys.argv[3])
        cmd = "cd mpi2 && make && mpirun -n {} decide {}".format(n, "{}")
    elif (sys.argv[1] == 'omp'):
        cmd = "cd omp && make && ./decide {}"
    elif (sys.argv[1] == 'omp_ab'):
        cmd = "cd omp_ab && make && ./decide {}"
    elif (sys.argv[1] == 'mpi_omp'):
        n = 4
        if len(sys.argv) >= 4:
            n = int(sys.argv[3])
        cmd = "cd mpi_omp && make && mpirun -n {} decide {}".format(n, "{}")

cmd = cmd.format(input)
print(cmd)
os.system(cmd)
