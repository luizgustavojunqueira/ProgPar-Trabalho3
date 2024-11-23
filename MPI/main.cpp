#include "./lib/cliques.hpp"
#include <iostream>
#include <sys/time.h>
#include <thread>
#include <time.h>

double read_timer() {
  static int initialized = 0;
  static struct timeval start;
  struct timeval end;
  if (!initialized) {
    gettimeofday(&start, NULL);
    initialized = 1;
  }
  gettimeofday(&end, NULL);
  return (end.tv_sec - start.tv_sec) + 1.0e-6 * (end.tv_usec - start.tv_usec);
}

double start_time, end_time; /* start and end times */

int main(int argc, char *argv[]) {

  if (argc < 4) {
    cout << "Usage: ./main <filename> <k> <alg>" << endl;
    return 1;
  }

  string filename = argv[1];
  int k = stoi(argv[2]);
  int alg = stoi(argv[3]);

  return 0;
}
