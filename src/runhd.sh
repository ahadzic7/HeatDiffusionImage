#!/bin/sh

g++ HeatDiffusion.cpp -lmsmpi -Ofast -std=c++17 -march=native -o mpi

mpiexec -np $1 ./mpi "C:\Users\Armin\Desktop\hw2_mpi_public\instances\test.txt" "C:\Users\Armin\Desktop\hw2_mpi_public\results\test.txt"
