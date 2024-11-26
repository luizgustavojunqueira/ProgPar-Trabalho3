#include "cliques.hpp"

void readGraph(string filename, vector<vector<int>> &adj_list, int &num_vertices){
  // Abre o arquivo de edges
  ifstream file(filename);
  map<int, int> vertex_map;
  vector<int> vertices;
  num_vertices = 0;

  int v, w;
  // Le uma aresta do arquivo
  while (file >> v >> w) {

    // Caso o vertice v nao esteja no map, adiciona ele
    if (vertex_map.find(v) == vertex_map.end()) {
      // Adiciona o vertice no vetor de vertices
      vertices.push_back(num_vertices);

      // Adiciona o vertice no map
      vertex_map[v] = num_vertices++;
    }

    // Caso o vertice w nao esteja no map, adiciona ele
    if (vertex_map.find(w) == vertex_map.end()) {
      // Adiciona o vertice no vetor de vertices
      vertices.push_back(num_vertices);
      // Adiciona o vertice no map
      vertex_map[w] = num_vertices++;
    }

    // Adiciona a aresta no grafo (lista de adjacencia)
    addEdge(vertex_map[v], vertex_map[w], adj_list);
  }

  // Ordena as listas de adjacencia
  for (int i = 0; i < num_vertices; i++) {
    sort(adj_list[i].begin(), adj_list[i].end());
  }

  file.close();
}

void addEdge(long unsigned int v, long unsigned int u, vector<vector<int>> &adj_list){

  // Como os vértices são indexados de 0 a n-1, o tamanho do vetor de adjacencia
  // é o maior vértice + 1

  long unsigned int size = adj_list.size();

  // Caso o vertice v nao exista, adiciona a lista de adjacencia dele
  if (size <= v) {
    adj_list.push_back(vector<int>());
  }

  // Caso o vertice u nao exista, adiciona a lista de adjacencia dele
  if (size <= u) {
    adj_list.push_back(vector<int>());
  }

  // Adiciona o vértice u na lista de adjacencia de v
  if (find(adj_list[v].begin(), adj_list[v].end(), u) ==
      adj_list[v].end()) {
    adj_list[v].push_back(u);
  }

  // Adiciona o vértice v na lista de adjacencia de u
  if (find(adj_list[u].begin(), adj_list[u].end(), v) ==
      adj_list[u].end()) {
    adj_list[u].push_back(v);
  }
}

int isNeighbor(int v, int vizinho, vector<vector<int>> &adj_list){
  // Como as listas de adjacencia estão ordenadas, é possível fazer uma busca
  // binária
  if (binary_search(adj_list[v].begin(), adj_list[v].end(), vizinho)) {
    return 1;
  }
  return 0;
}

int connectToAll(vector<int> clique, int v, vector<vector<int>> &adj_list){
  for (int vertex : clique) {
    if (isNeighbor(vertex, v, adj_list) == 0) {
      return 0;
    }
  }
  return 1;
}

int isInClique(vector<int> clique, int v) {
  for (int vertex : clique) {
    if (vertex == v) {
      return 1;
    }
  }
  return 0;
}

int formsNewClique(vector<int> clique, int v, vector<vector<int>> &adj_list) {
  // Para formar uma nova clique, o vértice v deve ser vizinho de todos os
  // vértices da clique e não deve estar na clique
  if (isInClique(clique, v) == 0 && connectToAll(clique, v, adj_list) == 1) {
    return 1;
  }
  return 0;
}
