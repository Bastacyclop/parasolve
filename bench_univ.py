#!/usr/bin/python3

import os
import sys
import subprocess

inputs = [
    '"7K//k1P/7p w"',
    '"///2kpK/7P w"',
    '"4k//4K/4P w"',
    '"4k//4K//4P w"',
    '"/ppp//PPP//7k//7K w"',
    '"4k//4K///4P w"',
    #'"7K//k1P/7p b"',
]

mpi_ns = [2, 4, 8, 16]
omp_ns = [2, 4, 8]
mpi_omp_ns = [(4, 2), (2, 4), (8, 2)]

args = sys.argv
if len(args) < 2:
    exit("expected bench name")

path = "bench_" + args[1]
if os.path.exists(path):
    print("'{}' already exists".format(path))
else:
    print("creating '{}' directory".format(path))
    os.mkdir(path)

def run(cmd):
    print(cmd)
    return int(subprocess.check_output(cmd, shell=True))

def run_seq(input):
    cmd = "./run_univ.py seq '{}' | grep \"execution time:\" | grep -oE \"[^ ]+$\""
    return run(cmd.format(input))

def run_mpi(input, n):
    cmd = "./run_univ.py mpi '{}' {} | grep \"execution time (0):\" | grep -oE \"[^ ]+$\""
    return run(cmd.format(input, n))

def run_omp(input, n):
    cmd = "OMP_NUM_THREADS={} ./run_univ.py omp '{}' | grep \"execution time:\" | grep -oE \"[^ ]+$\""
    return run(cmd.format(n, input))

def run_mpi_omp(input, mpi_n, omp_n):
    cmd = "OMP_NUM_THREADS={} ./run_univ.py mpi_omp '{}' {} | grep \"execution time (0):\" | grep -oE \"[^ ]+$\""
    return run(cmd.format(omp_n, input, mpi_n))

data_path = path + "/data"
if os.path.exists(data_path):
    print("'{}' already exists, skipping data collection".format(data_path))
else:
    print("collecting data into '{}'".format(data_path))
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

cmds_path = path + "/plot_cmds"
print("generating '{}'".format(cmds_path))
cmds = open(cmds_path, "w")
cmds.write("set terminal png size 800, 600\n")
cmds.write("set key outside\n")
cmds.write('set xlabel "fichier"\n')
time_plot = ' "data" u ($1+0.05*{}):{} w impulses title \'{} {}\','
efficiency_plot = ' "data" u ($1+0.05*{}):(($2/${})/{}) w impulses title \'{} {}\','
offset = 2
seq_time_plot = time_plot.format(0, offset, "seq", 1)
offset += 1

cmds.write('set output "mpi_time.png"\n')
cmds.write('set ylabel "temps d\'exécution (s)"\n')
cmds.write('plot')
cmds.write(seq_time_plot)
for index, n in enumerate(mpi_ns):
    col = offset + index
    cmds.write(time_plot.format(index + 1, col, "mpi", n))
cmds.write('\n')

cmds.write('set output "mpi_efficiency.png"\n')
cmds.write('set ylabel "efficacité"\n')
cmds.write('plot')
for index, n in enumerate(mpi_ns):
    col = offset + index
    cmds.write(efficiency_plot.format(index, col, n, "mpi", n))
cmds.write('\n')

offset += len(mpi_ns)

cmds.write('set output "omp_time.png"\n')
cmds.write('set ylabel "temps d\'exécution (s)"\n')
cmds.write('plot')
cmds.write(seq_time_plot)
for index, n in enumerate(omp_ns):
    col = offset + index
    cmds.write(time_plot.format(index + 1, col, "omp", n))
cmds.write('\n')

cmds.write('set output "omp_efficiency.png"\n')
cmds.write('set ylabel "efficacité"\n')
cmds.write('plot')
for index, n in enumerate(omp_ns):
    col = offset + index
    cmds.write(efficiency_plot.format(index, col, n, "omp", n))
cmds.write('\n')

offset += len(omp_ns)

cmds.write('set output "mpi_omp_time.png"\n')
cmds.write('set ylabel "temps d\'exécution (s)"\n')
cmds.write('plot')
cmds.write(seq_time_plot)
for index, n in enumerate(mpi_omp_ns):
    col = offset + index
    cmds.write(time_plot.format(index + 1, col, "mpi+omp", n))
cmds.write('\n')

cmds.write('set output "mpi_omp_efficiency.png"\n')
cmds.write('set ylabel "efficacité"\n')
cmds.write('plot')
for index, (m, n) in enumerate(mpi_omp_ns):
    col = offset + index
    procs = 1 + (m - 1) * n # the master does not spawn omp threads
    cmds.write(efficiency_plot.format(index, col, procs, "mpi+omp", (m, n)))
cmds.write('\n')

offset += len(mpi_omp_ns)

cmds.close()

print("plotting")
os.system("cd {} && gnuplot < plot_cmds".format(path))
