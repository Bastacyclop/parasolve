#!/usr/bin/python3

import os
import sys

input = '"4k//4K//4P w"'
if len(sys.argv) >= 3:
    input = sys.argv[2]

hostfile = ""
if os.path.exists("hostfile"):
    hostfile = "--hostfile ../hostfile --map-by node "

cmd = "cd seq && make && ./decide {}"
if len(sys.argv) >= 2:
    if (sys.argv[1] == 'seq'):
        pass
    elif (sys.argv[1][0:3] == 'mpi'):
        n = 4
        if len(sys.argv) >= 4:
            n = int(sys.argv[3])
        cmd = "cd {} && make && mpirun -n {} {} decide {}".format(sys.argv[1], n, hostfile, "{}")
    elif (sys.argv[1][0:3] == 'omp'):
        cmd = "cd {} && make && ./decide {}".format(sys.argv[1], "{}")
    else:
        exit("unknown method :s")

cmd = cmd.format(input)
print(cmd)
os.system(cmd)
