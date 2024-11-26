#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <thread>
#include <time.h>
#include <fstream>
#include <map>
#include <algorithm>

using namespace std;

#define MAX_CLIQUES 1000  // Limite de cliques por thread

// Função para verificar se todos os vértices na clique estão conectados entre si
__device__ bool verificaClique(int* clique, int clique_size, int* neigh_sizes, int* neighbor_lists, int num_vertices) {
    for (int i = 0; i < clique_size; i++) {
        for (int j = i + 1; j < clique_size; j++) {
            // A função verifica se os vértices clique[i] e clique[j] estão conectados, com base na lista de vizinhos
            int start = neigh_sizes[clique[i]];  // Get the starting position of neighbors for clique[i]
            int end = start + neigh_sizes[clique[i]];  // Get the end position of neighbors for clique[i]

            // Verifica se clique[j] está na lista de vizinhos de clique[i]
            bool found = false;
            for (int n = start; n < end; n++) {
                if (neighbor_lists[n] == clique[j]) {
                    found = true;
                    break;
                }
            }
            if (!found) return false;
        }
    }
    return true;
}

// Kernel CUDA para contagem de cliques
__global__ void contagem_de_cliques_kernel(int* neigh_sizes, int* neighbor_lists, int* d_cliques_count, int num_vertices, int k) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (thread_id >= num_vertices) return;

    // Variáveis locais para armazenar cliques temporárias
    int clique[MAX_CLIQUES];
    int clique_size = 1;
    clique[0] = thread_id;

    // Contagem de cliques locais
    int local_cliques = 0;

    // Expandir a clique tentando encontrar cliques de tamanho k
    for (int i = 0; i < neigh_sizes[thread_id]; i++) {
        int neighbor = neighbor_lists[neigh_sizes[thread_id] + i];

        clique[clique_size] = neighbor;
        clique_size++;

        // Se atingiu o tamanho k, verifica se é uma clique válida
        if (clique_size == k) {
            if (verificaClique(clique, clique_size, neigh_sizes, neighbor_lists, num_vertices)) {
                local_cliques++;
            }
            clique_size--;  // Desfaz a expansão para a próxima iteração
        }
    }

    // Incrementa a contagem de cliques globalmente de forma atômica
    atomicAdd(d_cliques_count, local_cliques);
}



void add_edge(long unsigned int v, long unsigned int u, vector<vector<int>> &adj_list) {

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


int main(int argc, char **argv) {
  if (argc < 3) {
    cout << "Usage: ./main <filename> <k>" << endl;
    return 1;
  }

  string filename = argv[1];
  int k = atoi(argv[2]);

  int num_vertices = 0;

  vector<int> vertices;
  vector<vector<int>> adj_list;

  ifstream file(filename);
  map<int, int> vertex_map;

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
    add_edge(vertex_map[v], vertex_map[w], adj_list);
  }


  // Ordena as listas de adjacencia
  for (int i = 0; i <num_vertices; i++) {
    sort(adj_list[i].begin(), adj_list[i].end());
  }

  cout << "Graph initialized" << endl;


  int **adj_matrix;

  adj_matrix = (int **)malloc(num_vertices * sizeof(int *));

  for (int i = 0; i < num_vertices; i++) {
    adj_matrix[i] = (int *)malloc(adj_list[i].size() * sizeof(int));
  }

  cout << "Graph matrix initialized" << endl;

  // Alocação de memória para a matriz de adjacência na GPU
  int* d_adj;
  int* d_cliques_count;
  int h_cliques_count = 0;

  cudaMalloc((void**)&d_adj, num_vertices * num_vertices * sizeof(int));
  cudaMalloc((void**)&d_cliques_count, sizeof(int));

  // Copiar a matriz de adjacência da memória do host para a memória da GPU
  cudaMemcpy(d_adj, adj_matrix, num_vertices * num_vertices * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_cliques_count, &h_cliques_count, sizeof(int), cudaMemcpyHostToDevice);

  cout << "Graph matrix copied to GPU" << endl;

  // Configuração do kernel
  int block_size = 256;
  int num_blocks = (num_vertices + block_size - 1) / block_size;

  // Lançar o kernel CUDA
  contagem_de_cliques_kernel<<<num_blocks, block_size>>>(d_adj, d_cliques_count, num_vertices, k);

  // Sincronizar a GPU
  cudaDeviceSynchronize();

  // Copiar o resultado da contagem de cliques de volta para o host
  cudaMemcpy(&h_cliques_count, d_cliques_count, sizeof(int), cudaMemcpyDeviceToHost);

  std::cout << "Número de cliques de tamanho " << k << ": " << h_cliques_count << std::endl;

  // Liberar memória na GPU
  cudaFree(d_adj);
  cudaFree(d_cliques_count);

  return 0;
}

