#include <iostream>
#include <sys/time.h>
#include <thread>
#include <time.h>
#include <vector>
#include <fstream>
#include <map>
#include <algorithm>

using namespace std;

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

__global__ void countCliques(int **adj, int *fila, int *contagem, int k, int num_vertices){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (idx >= num_vertices) return;

  int *clique_inicial = (int *)malloc(sizeof(int));
  int **cliques = (int **)malloc(sizeof(int *));
  int count = 0;
  int num_cliques = 1;

  clique_inicial[0] = fila[idx];
  cliques[0] = clique_inicial;

  int tamanho_da_fila = sizeof(fila) / sizeof(fila[0]);

  printf(" 1Thread %d. Vertice inicial: %d. Tamanho da fila: %d. Vertice: %d. Num cliques: %d. \n", idx, fila[idx], sizeof(fila), fila[idx], num_cliques);

  while(num_cliques > 0){
    int *clique = cliques[num_cliques - 1];

    printf(" 2Thread %d. Vertice inicial: %d. Tamanho da fila: %d. Vertice: %d. Num cliques: %d. \n", idx, fila[idx], sizeof(fila), clique[0], num_cliques);
    num_cliques--;

    int tamanho_clique = sizeof(*clique) / sizeof(int);

    printf(" 3Thread %d. Tamanho do clique: %d\n", idx, tamanho_clique);

    if(tamanho_clique == k){
      printf(" 4Thread %d. Clique de tamanho %d encontrado. Vertice inicial: %d. Tamanho da fila: %d. Vertice: %d. Num cliques: %d. \n", idx, k, fila[idx],sizeof(fila), clique[0], num_cliques);
      count++;
      continue;
    }

    int ultimo_vertice = clique[tamanho_clique - 1];

    printf(" 5Thread %d. Ultimo vertice: %d\n", idx, ultimo_vertice);

    for(int i = 0; i < tamanho_clique; i++){
      int vertice = clique[i];

      printf(" 6Thread %d. Vertice: %d\n", idx, vertice);

      // Get the size of the neighborhood of the vertex

      int tamanho_vizinhanca = sizeof(*adj[vertice]) / sizeof(adj[vertice][0]);

      printf(" 7Thread %d. Tamanho da vizinhança: %d. Do vertice %d.\n", idx, tamanho_vizinhanca, vertice);

      for(int j = 0; j < tamanho_vizinhanca; j++){
        int vizinho = adj[vertice][j];

        if(vizinho > ultimo_vertice){

          bool is_in_clique = false;

          for(int l = 0; l < tamanho_clique; l++){
            if(clique[l] == vizinho){
              is_in_clique = true;
              break;
            }
          }

          if(!is_in_clique){

            int forms_clique = true;

            for(int l = 0; l < tamanho_clique; l++){
              int vertice_clique = clique[l];

              int tamanho_vizinhanca_clique = sizeof(*adj[vertice_clique]) / sizeof(int);

              bool is_neighbor = false;

              for(int m = 0; m < tamanho_vizinhanca_clique; m++){
                if(adj[vertice_clique][m] == vizinho){
                  is_neighbor = true;
                  break;
                }
              }

              if(!is_neighbor){
                forms_clique = false;
                break;
              }
            }

            if(forms_clique){
              int *nova_clique = (int *)malloc((tamanho_clique + 1) * sizeof(int));

              for(int l = 0; l < tamanho_clique; l++){
                nova_clique[l] = clique[l];
              }

              nova_clique[tamanho_clique] = vizinho;

              num_cliques++;

              int ** novo_cliques = (int **)malloc((num_cliques + 1) * sizeof(int *));
              for(int l = 0; l < num_cliques; l++){
                novo_cliques[l] = cliques[l];
              }

              novo_cliques[num_cliques] = nova_clique;

              free(cliques);

              cliques = novo_cliques;

            }
          }
        }
      }
    }
  }

  atomicAdd(contagem, count);

  for(int i = 0; i < num_cliques; i++){
    free(cliques[i]);
  }

  free(cliques);
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

int main(int argc, char *argv[]) {

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

  int *fila_cliques;
  int **adj_listArr;

  fila_cliques = (int *)malloc(num_vertices * sizeof(int));
  adj_listArr = (int **)malloc(num_vertices * sizeof(int *));

  for (int i = 0; i < num_vertices; i++) {
    fila_cliques[i] = vertices[i];
    adj_listArr[i] = (int *)malloc(adj_list[i].size() * sizeof(int));
  }

  cout << "Transformed adj_list vector to array" << endl;

  int *device_fila_cliques;
  int **device_adj_list;
  int *device_row_ptr[num_vertices];

  cout << "Starting cudaMalloc" << endl;

  // Aloca memória para cada linha
  for(int i = 0; i < num_vertices; i++){
    cudaMalloc((void**)&device_row_ptr[i], adj_list[i].size() * sizeof(int));

    cout << "Copying row " << i << " to device" << endl;

    cudaMemcpy(device_row_ptr[i], adj_listArr[i], adj_list[i].size() * sizeof(int), cudaMemcpyHostToDevice);
  }

  cout << "Finished copying rows to device" << endl;

  cudaMalloc((void **)&device_adj_list, num_vertices * sizeof(int *));

  cudaMemcpy(device_adj_list, device_row_ptr, num_vertices * sizeof(int *), cudaMemcpyHostToDevice);

  cout << "Finished copying adj_list to device" << endl;

  cout << "Copying fila_cliques to device" << endl;

  cudaMalloc(&device_fila_cliques, num_vertices * sizeof(int));

  cudaMemcpy(device_fila_cliques, fila_cliques, num_vertices * sizeof(int), cudaMemcpyHostToDevice);

  cout << "Finished copying fila_cliques to device" << endl;

  int *d_contagem;
  int contagem = 0;


  cudaMalloc(&d_contagem, sizeof(int));
  cudaMemcpy(d_contagem, &contagem, sizeof(int), cudaMemcpyHostToDevice);


  cout << "Starting kernel" << endl;
  start_time = read_timer();
  
  countCliques<<<(num_vertices + 255) / 256, 256>>>(device_adj_list, device_fila_cliques, &contagem, k, num_vertices);
  cudaDeviceSynchronize();
  
  end_time = read_timer();

  cudaMemcpy(&contagem, d_contagem, sizeof(int), cudaMemcpyDeviceToHost);

  cout << "Finished kernel" << endl;

  cout << "Tempo de execução: " << end_time - start_time << "s" << endl;

  cout << "Número de cliques de tamanho " << k << ": " << contagem << endl;

  // for (int i = 0; i < num_vertices; i++) {
  //   free(fila_cliques[i]);
  //   free(adj_list[i]);
  // }
  //
  // free(fila_cliques);
  // free(adj_list);
  //
  // cudaFree(d_fila_cliques);
  // cudaFree(d_adj_list);
  // cudaFree(d_contagem);

  return 0;
}
