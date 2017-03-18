#!/bin/python

import os
import sys
import subprocess

inputs = [
    '"7K//k1P/7p w"',
    '"///2kpK/7P w"',
    #'"4k//4K/4P w"',
    #'"4k//4K//4P w"',
    #'"/ppp//PPP//7k//7K w"',
    #'"4k//4K///4P w"',
    #'"7K//k1P/7p b"',
]

args = sys.argv
if len(args) < 2:
    exit("expected bench name")

path = "bench_" + args[1]
if not os.path.exists(path):
    os.mkdir(path)

def run(cmd):
    print(cmd)
    return int(subprocess.run(cmd, stdout=subprocess.PIPE, shell=True).stdout)

def run_seq(input):
    cmd = "./run.py seq '{}' | grep \"execution time:\" | grep -oE \"[^ ]+$\""
    return run(cmd.format(input))

def run_mpi(input, n):
    cmd = "./run.py mpi '{}' {} | grep \"execution time (0):\" | grep -oE \"[^ ]+$\""
    return run(cmd.format(input, n))

def run_omp(input, n):
    cmd = "OMP_NUM_THREADS={} ./run.py omp '{}' | grep \"execution time:\" | grep -oE \"[^ ]+$\""
    return run(cmd.format(n, input))

def run_mpi_omp(input, mpi_n, omp_n):
    cmd = "OMP_NUM_THREADS={} ./run.py mpi_omp '{}' {} | grep \"execution time (0):\" | grep -oE \"[^ ]+$\""
    return run(cmd.format(omp_n, input, mpi_n))

mpi_ns = [2, 4]
omp_ns = [2, 4]
mpi_omp_ns = [(2, 2)]

"""
data = open(path + "/data", "w")
for index, i in enumerate(inputs):
    data.write(str(index))
    data.write(" {}".format(run_seq(i)))
    for n in mpi_ns:
        data.write(" {}".format(run_mpi(i, n)))
    for n in omp_ns:
        data.write(" {}".format(run_omp(i, n)))
    for (m, n) in mpi_omp_ns:
        data.write(" {}".format(run_mpi_omp(i, m, n)))
    data.write("\n")
data.close()
"""

cmds = open(path + "/plot_cmds", "w")
cmds.write("set terminal png size 800, 600\n")
offset = 1

cmds.write('set output "mpi.png"\n')
cmds.write('set xlabel "fichier"\n')
cmds.write('set ylabel "temps d\'exécution (s)"\n')
cmds.write('plot')
for index, n in enumerate(mpi_ns):
    col = offset + index
    cmds.write(' "data" u 1:{} w lines title \'mpi, n={}\','.format(col, n))
cmds.write('\n')
offset += len(mpi_ns)

cmds.write('set output "omp.png"\n')
cmds.write('plot')
for index, n in enumerate(omp_ns):
    col = offset + index
    cmds.write(' "data" u 1:{} w lines title \'omp, n={}\','.format(col, n))
cmds.write('\n')
offset += len(omp_ns)

cmds.write('set output "mpi_omp.png"\n')
cmds.write('plot')
for index, (m, n) in enumerate(mpi_omp_ns):
    col = offset + index
    cmds.write(' "data" u 1:{} w lines title \'(mpi, omp): ({}, {})\','.format(col, m, n))
cmds.write('\n')
offset += len(mpi_omp_ns)

cmds.close()

os.system("cd {} && gnuplot < plot_cmds".format(path))
