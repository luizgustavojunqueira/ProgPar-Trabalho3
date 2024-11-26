#include <iostream>
#include <vector>
#include <cuda_runtime.h>

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

int main() {
    int num_vertices = 8;  // Exemplo de número de vértices
    int k = 3;  // Tamanho da clique a ser procurado

    // Matriz de adjacência representada por listas de vizinhos (exemplo de grafo)
    std::vector<std::vector<int>> adj_list = {
        {1, 2},
        {0, 2, 3},
        {0, 1, 3, 4},
        {1, 2, 4, 5},
        {2, 3, 5, 6},
        {3, 4, 6, 7},
        {4, 5, 7},
        {5, 6}
    };

    // Alocação de memória para as listas de vizinhos e seus tamanhos na GPU
    int* d_neigh_sizes;
    int* d_neighbor_lists;
    int* d_cliques_count;
    int h_cliques_count = 0;
    
    std::vector<int> neigh_sizes(num_vertices);
    std::vector<int> neighbor_lists;
    
    // Preencher as listas de vizinhos e tamanhos
    for (int i = 0; i < num_vertices; i++) {
        neigh_sizes[i] = adj_list[i].size();
        neighbor_lists.insert(neighbor_lists.end(), adj_list[i].begin(), adj_list[i].end());
    }

    // Alocar memória na GPU
    cudaMalloc((void**)&d_neigh_sizes, num_vertices * sizeof(int));
    cudaMalloc((void**)&d_neighbor_lists, neighbor_lists.size() * sizeof(int));
    cudaMalloc((void**)&d_cliques_count, sizeof(int));

    // Copiar os dados do host para a memória da GPU
    cudaMemcpy(d_neigh_sizes, neigh_sizes.data(), num_vertices * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_neighbor_lists, neighbor_lists.data(), neighbor_lists.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cliques_count, &h_cliques_count, sizeof(int), cudaMemcpyHostToDevice);

    // Configuração do kernel
    int block_size = 256;
    int num_blocks = (num_vertices + block_size - 1) / block_size;

    // Lançar o kernel CUDA
    contagem_de_cliques_kernel<<<num_blocks, block_size>>>(d_neigh_sizes, d_neighbor_lists, d_cliques_count, num_vertices, k);

    // Sincronizar a GPU
    cudaDeviceSynchronize();

    // Copiar o resultado da contagem de cliques de volta para o host
    cudaMemcpy(&h_cliques_count, d_cliques_count, sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Número de cliques de tamanho " << k << ": " << h_cliques_count << std::endl;

    // Liberar memória na GPU
    cudaFree(d_neigh_sizes);
    cudaFree(d_neighbor_lists);
    cudaFree(d_cliques_count);

    return 0;
}

