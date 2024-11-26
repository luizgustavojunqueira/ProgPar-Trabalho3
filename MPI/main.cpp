#include "./lib/cliques.hpp"
#include <iostream>
#include <sys/time.h>
#include <thread>
#include <time.h>
#include <mpi.h>


int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);
  double tini = MPI_Wtime() ;
  
  if (argc < 4) {
    cout << "Usage: ./main <filename> <k> <alg>" << endl;
    return 1;
  }

  string filename = argv[1];
  int k = stoi(argv[2]);
  int alg = stoi(argv[3]);
  double tfim = MPI_Wtime();
  printf("Tempo de execução=%f segundos\n", tfim - tini);
  MPI_Finalize();

  return 0;
}
