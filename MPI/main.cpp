#include "cliques.hpp"

int main(int argc, char *argv[]) {

  if (argc < 3) {
    cout << "Usage: ./main <filename> <k>" << endl;
    return 1;
  }

  int k = stoi(argv[2]);
  string filename = argv[1];

  int idProcesso, nProcessos;

  double start, end;

  MPI_Init(&argc, &argv); 

  MPI_Comm_rank(MPI_COMM_WORLD, &idProcesso);
  MPI_Comm_size(MPI_COMM_WORLD, &nProcessos);

  if(idProcesso == 0){
    vector<vector<int>> adj_list;
    int num_vertices;

    readGraph(filename, adj_list, num_vertices);

    vector<int> flat_adj_list;
    vector<int> row_sizes;

    for (int i = 0; i < num_vertices; i++) {
      row_sizes.push_back(adj_list[i].size());
      for (int j = 0; j < adj_list[i].size(); j++) {
        flat_adj_list.push_back(adj_list[i][j]);
      }
    }

    long unsigned int row_sizes_size = row_sizes.size();
    long unsigned int flat_adj_list_size = flat_adj_list.size();

    for(int i = 1; i < nProcessos; i++){
      MPI_Send(&row_sizes_size, 1, MPI_UNSIGNED_LONG, i, 0, MPI_COMM_WORLD);
      MPI_Send(&flat_adj_list_size, 1, MPI_UNSIGNED_LONG, i, 0, MPI_COMM_WORLD);
      MPI_Send(row_sizes.data(), row_sizes.size(), MPI_INT, i, 0, MPI_COMM_WORLD);
      MPI_Send(flat_adj_list.data(), flat_adj_list.size(), MPI_INT, i, 0, MPI_COMM_WORLD);
    }

    queue<int> fila;
    int count = 0;

    for(int i = 0; i < num_vertices; i++){
      fila.push(i);
    }

    cout << "Fila iniciada, começando a distribuir trabalho e contando tempo" << endl;

    start = MPI_Wtime();

    while(fila.size() > 0){
      for(int i = 1; i < nProcessos; i++){
        int clique_count;
        MPI_Recv(&clique_count, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        count += clique_count;

        if(fila.size() > 0){
          int v = fila.front();
          fila.pop();
          MPI_Send(&v, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }else{
          int v = -1;
          MPI_Send(&v, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
      }
    }

    int v = -1;

    for(int i = 1; i < nProcessos; i++){
      MPI_Send(&v, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
    }

    cout << "Total number of cliques: " << count << endl;

    end = MPI_Wtime();

  cout << "Time: " << end - start << "s" << endl;

  }else{

    long unsigned int row_sizes_size;
    long unsigned int flat_adj_list_size;

    MPI_Recv(&row_sizes_size, 1, MPI_UNSIGNED_LONG, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(&flat_adj_list_size, 1, MPI_UNSIGNED_LONG, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    vector<int> row_sizes(row_sizes_size);

    MPI_Recv(row_sizes.data(), row_sizes.size(), MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    vector<int> flat_adj_list(flat_adj_list_size);

    MPI_Recv(flat_adj_list.data(), flat_adj_list.size(), MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    vector<vector<int>> adj_list;

    int index = 0;

    for(int i = 0; i < row_sizes.size(); i++){
      vector<int> row;
      for(int j = 0; j < row_sizes[i]; j++){
        row.push_back(flat_adj_list[index]);
        index++;
      }
      adj_list.push_back(row);
    }

    int clique_count = 0;

    while(true){
      int v;
      MPI_Send(&clique_count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
      MPI_Recv(&v, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      if(v == -1){
        break;
      }

      set<vector<int>> cliques;
      vector<int> clique;

      clique.push_back(v);
      cliques.insert(clique);

      clique_count = 0;

      while (!cliques.empty()) {

        vector<int> clique_atual = *cliques.begin();
        cliques.erase(clique_atual);

        if (clique_atual.size() == k) {
          clique_count += 1;
          continue;
        }

        int last_vertex = clique_atual.back();

        for (int v : clique_atual) {

          for (int vizinho : adj_list[v]) {
            if (vizinho > last_vertex &&
              formsNewClique(clique_atual, vizinho, adj_list)) {
              vector<int> nova_clique = clique_atual;
              nova_clique.push_back(vizinho);

              cliques.insert(nova_clique);
            }
          }
        }
      }
    }
  }

  MPI_Finalize(); 

  return 0;
}
