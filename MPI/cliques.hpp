#include <set>
#include <string>
#include <vector>
#include <queue>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <mpi.h>
#include <sys/time.h>

using namespace std;

void readGraph(string filename, vector<vector<int>> &adj_list, int &num_vertices);
void addEdge(long unsigned int v, long unsigned int u, vector<vector<int>> &adj_list);
int isNeighbor(int v, int vizinho, vector<vector<int>> &adj_list);
int connectToAll(vector<int> clique, int v, vector<vector<int>> &adj_list);
int isInClique(vector<int> clique, int v);
int formsNewClique(vector<int> clique, int v, vector<vector<int>> &adj_list);
