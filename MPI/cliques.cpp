#include "cliques.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <omp.h>
#include <imp.h>
#define MAX 100
//mpi code 

int idProcesso, nProcessos, i;
char msg[MAX];
MPI_Status status;
MPI_Init(&argc, &argv);


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

  // Ordena as listas de adjacencia
  for (int i = 0; i < this->num_vertices; i++) {
    sort(this->adj_list[i].begin(), this->adj_list[i].end());
  }

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

// Verifica se um vértice é vizinho de outro
int Graph::isNeighbor(int v, int vizinho) {
  // Como as listas de adjacencia estão ordenadas, é possível fazer uma busca
  // binária
  if (binary_search(this->adj_list[v].begin(), this->adj_list[v].end(),
                    vizinho)) {
    return 1;
  }
  return 0;
}

// Verifica se um vértice é vizinho de todos os vértices de uma clique
int Graph::connectToAll(vector<int> clique, int v) {
  // Para cada vértice da clique
  // Verifica se o vértice v é vizinho
  for (int vertex : clique) {
    if (this->isNeighbor(vertex, v) == 0) {
      return 0;
    }
  }
  return 1;
}

// Verifica se um vértice está em uma clique
int Graph::isInClique(vector<int> clique, int v) {
  for (int vertex : clique) {
    if (vertex == v) {
      return 1;
    }
  }
  return 0;
}

// Verifica se um vértice pode formar uma nova clique
int Graph::formsNewClique(vector<int> clique, int v) {
  // Para formar uma nova clique, o vértice v deve ser vizinho de todos os
  // vértices da clique e não deve estar na clique
  if (this->isInClique(clique, v) == 0 && this->connectToAll(clique, v) == 1) {
    return 1;
  }
  return 0;
}

unsigned int Graph::countCliquesStatic(unsigned long k) {

  unsigned int count = 0;
  int num_threads = omp_get_max_threads();

  // Vetor de sets de cliques, um para cada thread
  vector<set<vector<int>>> thread_cliques(num_threads);

  // Inicializa cada set com cliques de tamanho 1, dividindo os vértices entre
  // threads
#pragma omp parallel for schedule(static) reduction(+ : count)
  for (int v = 0; v < this->num_vertices; v++) {
    int thread_id = omp_get_thread_num();
    thread_cliques[thread_id].insert({v});
    while (!thread_cliques[thread_id].empty()) {
      std::vector<int> clique_atual = *thread_cliques[thread_id].begin();
      thread_cliques[thread_id].erase(thread_cliques[thread_id].begin());

      if (clique_atual.size() == k) {
        count += 1;
        continue;
      }

      int last_vertex = clique_atual.back();

      for (int v : clique_atual) {
        for (int vizinho : this->adj_list[v]) {
          if (vizinho > last_vertex &&
              this->formsNewClique(clique_atual, vizinho)) {
            std::vector<int> nova_clique = clique_atual;
            nova_clique.push_back(vizinho);
            thread_cliques[thread_id].insert(nova_clique);
          }
        }
      }
    }
  }

  return count;
}

unsigned int Graph::countCliquesDynamic(unsigned long k, int chunk) {

  unsigned int count = 0;
  int num_threads = omp_get_max_threads();

  // Vetor de sets de cliques, um para cada thread
  vector<set<vector<int>>> thread_cliques(num_threads);

  // Inicializa cada set com cliques de tamanho 1, dividindo os vértices entre
  // threads
#pragma omp parallel for schedule(dynamic, chunk) reduction(+ : count)
  for (int v = 0; v < this->num_vertices; v++) {
    int thread_id = omp_get_thread_num();
    thread_cliques[thread_id].insert({v});
    while (!thread_cliques[thread_id].empty()) {
      std::vector<int> clique_atual = *thread_cliques[thread_id].begin();
      thread_cliques[thread_id].erase(thread_cliques[thread_id].begin());

      if (clique_atual.size() == k) {
        count += 1;
        continue;
      }

      int last_vertex = clique_atual.back();

      for (int v : clique_atual) {
        for (int vizinho : this->adj_list[v]) {
          if (vizinho > last_vertex &&
              this->formsNewClique(clique_atual, vizinho)) {
            std::vector<int> nova_clique = clique_atual;
            nova_clique.push_back(vizinho);
            thread_cliques[thread_id].insert(nova_clique);
          }
        }
      }
    }
  }

  return count;
}
unsigned int Graph::countCliquesGuided(unsigned long k) {

  unsigned int count = 0;
  int num_threads = omp_get_max_threads();

  // Vetor de sets de cliques, um para cada thread
  vector<set<vector<int>>> thread_cliques(num_threads);

  // Inicializa cada set com cliques de tamanho 1, dividindo os vértices entre
  // threads
#pragma omp parallel for schedule(guided) reduction(+ : count)
  for (int v = 0; v < this->num_vertices; v++) {
    int thread_id = omp_get_thread_num();
    thread_cliques[thread_id].insert({v});
    while (!thread_cliques[thread_id].empty()) {
      std::vector<int> clique_atual = *thread_cliques[thread_id].begin();
      thread_cliques[thread_id].erase(thread_cliques[thread_id].begin());

      if (clique_atual.size() == k) {
        count += 1;
        continue;
      }

      int last_vertex = clique_atual.back();

      for (int v : clique_atual) {
        for (int vizinho : this->adj_list[v]) {
          if (vizinho > last_vertex &&
              this->formsNewClique(clique_atual, vizinho)) {
            std::vector<int> nova_clique = clique_atual;
            nova_clique.push_back(vizinho);
            thread_cliques[thread_id].insert(nova_clique);
          }
        }
      }
    }
  }

  return count;
}

unsigned int Graph::countCliquesBalanceado(unsigned long k, int r) {
  unsigned int count = 0;
  int num_threads = omp_get_max_threads();
  vector<set<vector<int>>> thread_cliques(num_threads);
  vector<omp_lock_t> locks(num_threads);
  vector<int> counts(num_threads, 0);

  for (int i = 0; i < num_threads; i++) {
    omp_init_lock(&locks[i]);
  }

// Inicializa cada set com cliques de tamanho 1, dividindo os vértices entre
// threads
#pragma omp parallel for schedule(static)
  for (int v = 0; v < this->num_vertices; v++) {
    int thread_id = omp_get_thread_num();
    thread_cliques[thread_id].insert({v});
  }

#pragma omp parallel num_threads(num_threads)
  {
    int thread_id = omp_get_thread_num();

    while (true) {
      vector<int> clique_atual;

      // Lock na thread para acessar o clique
      omp_set_lock(&locks[thread_id]);
      if (!thread_cliques[thread_id].empty()) {
        clique_atual = *thread_cliques[thread_id].begin();
        thread_cliques[thread_id].erase(thread_cliques[thread_id].begin());
      }
      omp_unset_lock(&locks[thread_id]);

      if (!clique_atual.empty()) {
        if (clique_atual.size() == k) {
          counts[thread_id] += 1;
          continue;
        }

        int last_vertex = clique_atual.back();
        omp_set_lock(&locks[thread_id]);
        for (int v : clique_atual) {
          for (int vizinho : this->adj_list[v]) {
            if (vizinho > last_vertex &&
                this->formsNewClique(clique_atual, vizinho)) {
              vector<int> nova_clique = clique_atual;
              nova_clique.push_back(vizinho);

              thread_cliques[thread_id].insert(nova_clique);
            }
          }
        }
        omp_unset_lock(&locks[thread_id]);
      } else {
        bool allEmpty = true;

        // Tenta roubar cliques de outras threads
        for (int i = 0; i < num_threads; i++) {
          if (i != thread_id) {

            if (thread_cliques[i].empty()) {
              continue;
            }

            allEmpty = false;

            omp_set_lock(&locks[i]);
            if (thread_cliques[i].size() < r) {
              omp_unset_lock(&locks[i]);
              continue;
            }

            for (int j = 0; j < r; j++) {
              thread_cliques[thread_id].insert(*thread_cliques[i].rbegin());
              thread_cliques[i].erase(*thread_cliques[i].rbegin());
            }
            omp_unset_lock(&locks[i]);
            break;
          }
        }

        if (allEmpty) {
          break;
        }
      }
    }
  }

  // Soma dos resultados
  for (int i = 0; i < num_threads; i++) {
    count += counts[i];
    omp_destroy_lock(&locks[i]);
  }

  return count;
}
