all: mpi1 omp

mpi1:
	mkdir -p out
	mpicc -o out/lab2_mpi -Wall mpi_main.c

omp:
	mkdir -p out
	gcc-10 -o out/lab2_omp -Wall -fopenmp main.c

clean:
	rm -rf out

mpi1_run: mpi1
	mpirun -np 5 ./out/lab2_mpi

omp_run:
	./out/lab2_mpi