#include <set>
#include <string>
#include <vector>

#include <omp.h>

using namespace std;

class Graph {
public:
  Graph(string filename);
  void print();
  void print_edges();
  int num_vertices;
  vector<vector<int>> adj_list;
  int isNeighbor(int v, int w);
  int connectToAll(vector<int> clique, int v);
  int isInClique(vector<int> clique, int v);
  int formsNewClique(vector<int> clique, int v);

  unsigned int countCliquesStatic(unsigned long k);
  unsigned int countCliquesDynamic(unsigned long k, int chunk);
  unsigned int countCliquesGuided(unsigned long k);

  unsigned int countCliquesBalanceado(unsigned long k, int r);

private:
  vector<int> vertices;
  void add_edge(long unsigned int v, long unsigned int w);
};
