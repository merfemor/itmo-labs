all: mpi1 omp

mpi1:
	mkdir -p out
	mpicc -o out/lab2_mpi -Wall -DPRINT_DEBUG_MATRICES mpi_main.c

omp:
	mkdir -p out
	gcc-10 -o out/lab2_omp -Wall -fopenmp -DPRINT_DEBUG_MATRICES main.c

mpi1_release:
	mkdir -p out
	mpicc -o out/lab2_mpi -Wall -O3 mpi_main.c

omp_release:
	mkdir -p out
	gcc-10 -o out/lab2_omp -Wall -O3 -fopenmp main.c

clean:
	rm -rf out

mpi1_run: mpi1
	mpirun -np 5 ./out/lab2_mpi

omp_run: omp
	./out/lab2_omp

mpi1_run_release: mpi1_release
	mpirun -np 5 ./out/lab2_mpi

omp_run_release: omp_release
	./out/lab2_omp