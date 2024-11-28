#include "./cliques.cuh"

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

using namespace std;

__global__ void count_cliques(int *fila, int *fila_index, int *count, int *flat_adj_list_arr, int *offsets_arr, int *thread_mem, int *thread_mem_offsets, int num_thread_mem){
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  int mem_space = -1;

  for(int i = 0; i < num_thread_mem; i++){
    if(atomicCAS(&thread_mem[i], 0, 1) == 0){
      mem_space = i;
      break;
    }
  }

  if(mem_space == -1){
    printf("ERRO: Thread %d, não conseguiu espaço para suas cliques\n", thread_id);
  }

  int filaIndex = atomicAdd(fila_index, 1);
  int startVertex = fila[filaIndex];

  int clique[MAX_CLIQUE_SIZE];
  clique[0] = startVertex;

  int startCliquesMemIndex = thread_mem_offsets[thread_id] + 1;
  thread_mem[startCliquesMemIndex - 1] = 1;

  thread_mem[startCliquesMemIndex] = clique[0];
  int qntCliques = 1;
  int inicioCliques = 0;

  bool filled = false;

  while(qntCliques - inicioCliques >0){
    // Indice da primeira clique das cliques dessa thread
    int cliqueIndex = startCliquesMemIndex + inicioCliques * MAX_CLIQUE_SIZE;
    // Move o inicio das cliques para a próxima clique
    inicioCliques++;

    // Calcula o tamanho da clique atual
    int cliqueSize = 0;
    for(int i = 0; i < MAX_CLIQUE_SIZE; i++){
      if(thread_mem[cliqueIndex + i] != -1){
        cliqueSize++;
      }
    }

    // Se a clique atual já tem o tamanho máximo, incrementa o contador e passa para a próxima clique
    if(cliqueSize == MAX_CLIQUE_SIZE){
      atomicAdd(count, 1);
      continue;
    }
    
    // Pega o último vértice da clique atual
    int lastVertex = thread_mem[cliqueIndex + cliqueSize - 1];

    // Percorre os vértices da clique atual
    for(int i = 0; i < cliqueSize; i++){
      // Pega o vértice atual
      int vertexAtual = thread_mem[cliqueIndex + i];

      // Percorre os vizinhos do vértice atual
      for(int j = offsets_arr[vertexAtual]; j < offsets_arr[vertexAtual + 1]; j++){
        // Pega o vizinho
        int vizinho = flat_adj_list_arr[j];

        // Se o vizinho for maior que o último vértice da clique
        if(vizinho > lastVertex){

          // Verifica se o vizinho já está na clique
          bool isInClique = false;


          // Percorre os vértices da clique
          for(int k = 0; k < cliqueSize; k++){
            // Se o vizinho já está na clique
            if(thread_mem[cliqueIndex + k] == vizinho){
              isInClique = true;
              break;
            }
          }

          // Se o vizinho não está na clique
          if(!isInClique){

            // Verifica se o vizinho é vizinho de todos os vértices da clique
            bool ehVizinhoDeTodos = true;

            // Percorre os vértices da clique
            for(int k = 0; k < cliqueSize; k++){
              // Pega o vértice da clique
              int vertexClique = thread_mem[cliqueIndex + k];

              bool ehVizinho = false;

              // Percorre os vizinhos do vértice da clique
              for(int l = offsets_arr[vertexClique]; l < offsets_arr[vertexClique + 1]; l++){
                // Se o vizinho do vértice da clique for o vizinho
                if(flat_adj_list_arr[l] == vizinho){
                  ehVizinho = true;
                  break;
                }
              }

              // Se o vizinho não for vizinho de um dos vértices da clique, não é vizinho de todos
              if(!ehVizinho){
                ehVizinhoDeTodos = false;
                break;
              }
            }

            // Se o vizinho for vizinho de todos os vértices da clique
            if(ehVizinhoDeTodos){

              // Cria uma nova clique com o vizinho
              int newClique[MAX_CLIQUE_SIZE];
              for(int k = 0; k < cliqueSize; k++){
                newClique[k] = thread_mem[cliqueIndex + k];
              }
              newClique[cliqueSize] = vizinho;

              // add -1 to the end of the clique if it needs, considering MAX_CLIQUE_SIZE can change
              for(int k = cliqueSize + 1; k < MAX_CLIQUE_SIZE; k++){
                newClique[k] = -1;
              }

              // Verifica se a nova clique já existe
              bool cliqueJaExiste = false;

              // Percorre as cliques
              for(int k = 0; k < qntCliques; k++){
                // Pega a clique atual
                int cliqueAtual[MAX_CLIQUE_SIZE];
                for(int l = 0; l < MAX_CLIQUE_SIZE; l++){
                  cliqueAtual[l] = thread_mem[startCliquesMemIndex + k * MAX_CLIQUE_SIZE + l];
                }

                // Verifica se a nova clique é igual a clique atual
                bool saoIguais = true;
                for(int l = 0; l < MAX_CLIQUE_SIZE; l++){
                  if(newClique[l] != cliqueAtual[l]){
                    saoIguais = false;
                    break;
                  }
                }

                // Se a nova clique é igual a clique atual, a nova clique já existe
                if(saoIguais){
                  cliqueJaExiste = true;
                  break;
                }
              }

              if(!cliqueJaExiste){
                if(startCliquesMemIndex + qntCliques * MAX_CLIQUE_SIZE + MAX_CLIQUE_SIZE >= (startCliquesMemIndex + MAX_CLIQUES * MAX_CLIQUE_SIZE)){
                  printf("ERRO: Thread %d, encheu o seu espaço de cliques, parando para evitar de usar memória de outra thread\n", thread_id);
                  filled = true;
                  break;
                }
                // Adiciona a nova clique
                for(int k = 0; k < MAX_CLIQUE_SIZE; k++){
                  thread_mem[startCliquesMemIndex + qntCliques * MAX_CLIQUE_SIZE + k] = newClique[k];
                }
              }

              qntCliques++;
            }
          }
        }
        if(filled){
          break;
        }
      }

      if(filled){
        break;
      }
    }

    if(filled){
      break;
    }
  }

  for(int i = 0; i < MAX_CLIQUES * MAX_CLIQUE_SIZE; i++){
    thread_mem[mem_space + i] = -1;
  }

  thread_mem[mem_space] = 0;
}

int main(int argc, char *argv[]){

  if (argc < 2){
    cout << "Usage: " << argv[0] << " <input_file>" << endl;
    return 1;
  }

  cout << "Usando valor de MAX_CLIQUE_SIZE = " << MAX_CLIQUE_SIZE << endl;
  cout << "Usando valor de MAX_CLIQUES = " << MAX_CLIQUES << endl;
  cout << "Caso deseje alterar, altere os valores das constantes MAX_CLIQUE_SIZE e MAX_CLIQUES no arquivo cliques.cuh" << endl;

  string filename = argv[1];

  int num_vertices = 0;
  vector<int> vertices;
  vector<vector<int>> adj_list;

  readGraph(filename, adj_list, num_vertices);

  vector<int> flat_adj_list;
  vector<int> offsets;

  flatten(adj_list, flat_adj_list, offsets);

  int *flat_adj_list_arr = (int *)malloc(flat_adj_list.size() * sizeof(int));
  toArray(flat_adj_list, flat_adj_list_arr);

  int *offsets_arr = (int *)malloc(offsets.size() * sizeof(int));
  toArray(offsets, offsets_arr);

  // print adj list from array
  // for(int i = 0; i < num_vertices; i++){
  //   cout << i << ": ";
  //   for(int j = offsets_arr[i]; j < offsets_arr[i+1]; j++){
  //     cout << flat_adj_list_arr[j] << " ";
  //   }
  //   cout << endl;
  // }
  //

  int *fila_h = (int *)malloc(num_vertices * sizeof(int));

  for(int i = 0; i < num_vertices; i++){
    fila_h[i] = i;
  }

    // Alloc and copy fila_h to device
  int *fila_d;
  cudaMalloc((void **)&fila_d, num_vertices * sizeof(int));
  cudaMemcpy(fila_d, fila_h, num_vertices * sizeof(int), cudaMemcpyHostToDevice);

  int fila_index_h = 0;
  int *fila_index_d;
  cudaMalloc((void **)&fila_index_d, sizeof(int));
  cudaMemcpy(fila_index_d, &fila_index_h, sizeof(int), cudaMemcpyHostToDevice);

  int count_h = 0;
  int *count_d;
  cudaMalloc((void **)&count_d, sizeof(int));
  cudaMemcpy(count_d, &count_h, sizeof(int), cudaMemcpyHostToDevice);

  int *flat_adj_list_arr_d;
  cudaMalloc((void **)&flat_adj_list_arr_d, flat_adj_list.size() * sizeof(int));
  cudaMemcpy(flat_adj_list_arr_d, flat_adj_list_arr, flat_adj_list.size() * sizeof(int), cudaMemcpyHostToDevice);

  int *offsets_arr_d;
  cudaMalloc((void **)&offsets_arr_d, offsets.size() * sizeof(int));
  cudaMemcpy(offsets_arr_d, offsets_arr, offsets.size() * sizeof(int), cudaMemcpyHostToDevice);

  int block_size = 64;
  int amount_of_blocks = num_vertices / block_size;

  int num_threads = block_size * amount_of_blocks; // basicamente o número de vértices, ta em outra variável só pq o código é meu

  cout << "Number of threads: " << num_threads << endl;

  int num_thread_mem = num_threads * MAX_CLIQUES * MAX_CLIQUE_SIZE;

  int *thread_mem_offsets_h;

  thread_mem_offsets_h = (int *)malloc(num_threads * sizeof(int));

  for(int i = 0; i < num_threads; i++){
    thread_mem_offsets_h[i] = i * MAX_CLIQUES * MAX_CLIQUE_SIZE;
  }

  int *thread_mem_offsets_d;
  cudaMalloc((void **)&thread_mem_offsets_d, num_threads * sizeof(int));
  cudaMemcpy(thread_mem_offsets_d, thread_mem_offsets_h, num_threads * sizeof(int), cudaMemcpyHostToDevice);

  int *thread_mem_h = (int *)malloc(( num_threads + num_threads * MAX_CLIQUES * MAX_CLIQUE_SIZE ) * sizeof(int));

  memset(thread_mem_h, -1, ( num_threads + num_threads * MAX_CLIQUES * MAX_CLIQUE_SIZE ) * sizeof(int));

  for(int i = 0; i < num_threads; i++){
    thread_mem_h[thread_mem_offsets_h[i] + i] = 0;
  }

  int *thread_mem_d;
  // Aloca memória para as cliques
  cudaMalloc((void **)&thread_mem_d, ( num_threads + num_threads * MAX_CLIQUES * MAX_CLIQUE_SIZE ) * sizeof(int));
  cudaMemcpy(thread_mem_d, thread_mem_h, ( num_threads + num_threads * MAX_CLIQUES * MAX_CLIQUE_SIZE ) * sizeof(int), cudaMemcpyHostToDevice);

  start_time = read_timer();

  count_cliques<<<amount_of_blocks, block_size>>>(fila_d, fila_index_d, count_d, flat_adj_list_arr_d, offsets_arr_d, thread_mem_d, thread_mem_offsets_d, num_thread_mem);
  cudaDeviceSynchronize();

  end_time = read_timer();

  cudaMemcpy(&count_h, count_d, sizeof(int), cudaMemcpyDeviceToHost);

  int *cliques_h = (int *)malloc(num_threads * MAX_CLIQUES * MAX_CLIQUE_SIZE * sizeof(int));
  cudaMemcpy(cliques_h, thread_mem_d, num_threads * MAX_CLIQUES * MAX_CLIQUE_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

  cout << "Number of cliques: " << count_h << endl;
  cout << "Time: " << end_time - start_time << " seconds" << endl;

  free(flat_adj_list_arr);
  free(offsets_arr);
  free(fila_h);
  free(cliques_h);
  
  cudaFree(fila_d);
  cudaFree(fila_index_d);
  cudaFree(count_d);
  cudaFree(flat_adj_list_arr_d);
  cudaFree(offsets_arr_d);
  cudaFree(thread_mem_d);

  return 0;

}
