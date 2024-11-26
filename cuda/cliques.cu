#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>

Graph::Graph(string filename) {
  // Abre o arquivo de edges
  ifstream file(filename);
  map<int, int> vertex_map;
  int num_vertices = 0;

  int v, w;
  // Le uma aresta do arquivo
  while (file >> v >> w) {

    // Caso o vertice v nao esteja no map, adiciona ele
    if (vertex_map.find(v) == vertex_map.end()) {
      // Adiciona o vertice no vetor de vertices
      this->vertices.push_back(num_vertices);

      // Adiciona o vertice no map
      vertex_map[v] = num_vertices++;
    }

    // Caso o vertice w nao esteja no map, adiciona ele
    if (vertex_map.find(w) == vertex_map.end()) {
      // Adiciona o vertice no vetor de vertices
      this->vertices.push_back(num_vertices);
      // Adiciona o vertice no map
      vertex_map[w] = num_vertices++;
    }

    // Adiciona a aresta no grafo (lista de adjacencia)
    this->add_edge(vertex_map[v], vertex_map[w]);
  }

  this->num_vertices = num_vertices;

  // // Ordena as listas de adjacencia
  // for (int i = 0; i < this->num_vertices; i++) {
  //   sort(this->adj_list[i].begin(), this->adj_list[i].end());
  // }

  file.close();

}

// Adiciona uma aresta no grafo
void Graph::add_edge(long unsigned int v, long unsigned int u) {

  // Como os vértices são indexados de 0 a n-1, o tamanho do vetor de adjacencia
  // é o maior vértice + 1

  long unsigned int size = this->adj_list.size();

  // Caso o vertice v nao exista, adiciona a lista de adjacencia dele
  if (size <= v) {
    this->adj_list.push_back(vector<int>());
  }

  // Caso o vertice u nao exista, adiciona a lista de adjacencia dele
  if (size <= u) {
    this->adj_list.push_back(vector<int>());
  }

  // Adiciona o vértice u na lista de adjacencia de v
  if (find(this->adj_list[v].begin(), this->adj_list[v].end(), u) ==
      this->adj_list[v].end()) {
    this->adj_list[v].push_back(u);
  }

  // Adiciona o vértice v na lista de adjacencia de u
  if (find(this->adj_list[u].begin(), this->adj_list[u].end(), v) ==
      this->adj_list[u].end()) {
    this->adj_list[u].push_back(v);
  }
}

void Graph::copyToDevice(){
  cudaMalloc(&this->d_verticesArr, this->num_vertices * sizeof(int));
  cudaMalloc(&this->d_adj_list, this->num_vertices * sizeof(int*));

  cudaMemcpy(this->d_verticesArr, this->h_verticesArr, this->num_vertices * sizeof(int), cudaMemcpyHostToDevice);

  for (int i = 0; i < this->num_vertices; i++) {
    int size = this->adj_list[i].size();
    cudaMalloc(&this->d_adj_list[i], size * sizeof(int));
    cudaMemcpy(this->d_adj_list[i], this->h_adj_list[i], size * sizeof(int), cudaMemcpyHostToDevice);
  }
}

void Graph::startCliques(){
  this->h_cliques = new int*[this->num_vertices];
  for(int i = 0; i < this->num_vertices; i++){
    this->h_cliques[i] = new int[1];
    this->h_cliques[i][0] = this->h_verticesArr[i];
  }

  cudaMalloc(&this->d_cliques, this->num_vertices * sizeof(int*));
  for(int i = 0; i < this->num_vertices; i++){
    cudaMalloc(&this->d_cliques[i], sizeof(int));
    cudaMemcpy(this->d_cliques[i], this->h_cliques[i], sizeof(int), cudaMemcpyHostToDevice);
  }
}

void Graph::convertToArrays(){
  this->h_verticesArr = new int[this->num_vertices];
  this->h_adj_list = new int*[this->num_vertices];

  for (int i = 0; i < this->num_vertices; i++) {
    this->h_verticesArr[i] = this->vertices[i];
    int size = this->adj_list[i].size();
    this->h_adj_list[i] = new int[size];
    for (int j = 0; j < size; j++) {
      this->h_adj_list[i][j] = this->adj_list[i][j];
    }
  }
}


void Graph::countCliques(int k, int num_vertices, int *d_cliques, int *d_verticesArr, int **d_adj_list){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < num_vertices){
    
  }
}
