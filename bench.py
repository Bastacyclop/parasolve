#!/usr/bin/python3

import os
import sys
import subprocess

inputs_naive = [
    '"7K//k1P/7p w"',
    '"///2kpK/7P w"',
    '"4k//4K/4P w"',
    '"4k//4K//4P w"',
    '"/ppp//PPP//7k//7K w"',
    '"4k//4K///4P w"',
    '"7K//k1P/7p b"',
]

inputs_ab = [
    '"5k//2K1P b"',
    '"4k///4K///4P w"',
    '"/5p/4p/4PP/4K///k w"',
    '"/2k1K////3P/3P w"',
    '"/5p/4p/4P/4KP///k w"',
]

inputs_ab_tt = [
    '"K////2k3P/6P/6P/ w"',
    '"/k/3p/p2P1p/P2P1P///K w"',
    '"//pppk/5K//2P2P w"',
    '"4k//1p1p/p//1PPK w"',
    '"/6pp/5p/3k1PP/5K1P w"',
]

mpi_ns = [2, 4, 8, 16]
omp_ns = [2, 4, 8]
mpi_omp_ns = [(4, 2), (4, 4), (8, 4), (16, 8)]

args = sys.argv
if len(args) < 3:
    exit("expected bench and method name")

bench_path = "bench_" + args[1]
if os.path.exists(bench_path):
    print("'{}' already exists".format(bench_path))
else:
    print("creating '{}' directory".format(bench_path))
    os.mkdir(bench_path)

seq_path = "seq"
method = args[2]
inputs = None
method_path = method
if method in ["mpi", "mpi2"]:
    omp_ns = []
    mpi_omp_ns = []
    inputs = inputs_naive
elif method in ["omp"]:
    mpi_ns = []
    mpi_omp_ns = []
    inputs = inputs_naive
elif method in ["mpi_omp"]:
    mpi_ns = []
    omp_ns = []
    inputs = inputs_naive
elif method in ["omp_ab", "omp_ab_tt"]:
    mpi_ns = []
    mpi_omp_ns = []
    inputs = inputs_ab if method == "omp_ab" else inputs_ab_tt
    method_path = "omp_ab_tt"
else:
    exit("unknown method")

def find_flag(path, flag):
    return subprocess.check_output("grep '#define {}' {}/projet.h".format(flag, path), shell=True).decode("utf-8")

def print_flags(path):
    print("{}:".format(path))
    print(find_flag(path, "ALPHA_BETA_PRUNING"), end='')
    print(find_flag(path, "TRANSPOSITION_TABLE"), end='')

print("inputs: {}".format(inputs))
print("(mpi_ns, omp_ns, mpi_omp_ns): {}".format((mpi_ns, omp_ns, mpi_omp_ns)))
print_flags(seq_path)
print_flags(method_path)
if input("is this looking good ? (y/n) ") != "y":
    exit("see ya!")

def run(cmd):
    print(cmd)
    return int(subprocess.check_output(cmd, shell=True))

def run_seq(input):
    cmd = "./run.py seq '{}' | grep \"execution time:\" | grep -oE \"[^ ]+$\""
    return run(cmd.format(input))

def run_mpi(path, input, n):
    cmd = "./run.py {} '{}' {} | grep \"execution time (0):\" | grep -oE \"[^ ]+$\""
    return run(cmd.format(path, input, n))

def run_omp(path, input, n):
    cmd = "OMP_NUM_THREADS={} ./run.py {} '{}' | grep \"execution time:\" | grep -oE \"[^ ]+$\""
    return run(cmd.format(n, path, input))

def run_mpi_omp(path, input, mpi_n, omp_n):
    cmd = "OMP_NUM_THREADS={} ./run.py {} '{}' {} | grep \"execution time (0):\" | grep -oE \"[^ ]+$\""
    return run(cmd.format(omp_n, path, input, mpi_n))

def write_data(data, index, processors, time):
    data.write("{} {} {}\n".format(index, processors, time))

data_path = bench_path + "/" + method
if os.path.exists(data_path):
    print("'{}' already exists, skipping data collection".format(data_path))
else:
    print("collecting data into '{}'".format(data_path))
    data = open(data_path, "w")
    for index, i in enumerate(inputs):
        write_data(data, index, 1, run_seq(i))
        for m in mpi_ns:
            write_data(data, index, m, run_mpi(method_path, i, m))
        for n in omp_ns:
            write_data(data, index, n, run_omp(method_path, i, n))
        for (m, n) in mpi_omp_ns:
            p = 1 + (m - 1) * n
            write_data(data, index, p, run_mpi_omp(method_path, i, m, n))
    data.close()
