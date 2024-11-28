#!/bin/bash

#compilar o código
make

echo " " >> cliques.results
echo "==============================">> cliques.results
date +"%d/%m/%Y %H:%M:%S">> cliques.results
echo "Executando o código para o dataset citeseer"

echo "citeseer" >> cliques.results

for k in 3 4 5 6
do
  echo "Executando para k = $k"
  mpirun -np 4 ./main ../graph_datasets/citeseer.edgelist $k >> cliques.results
done


echo " " >> cliques.results
echo "ca_astroph" >> cliques.results

for k in 3 4 5
do
  echo "Executando para k = $k"
  mpirun -np 4 ./main ../graph_datasets/ca_astroph.edgelist $k >> cliques.results
done

echo " " >> cliques.results
echo "dblp" >> cliques.results

for k in 3 4 5
do
  echo "Executando para k = $k"
  mpirun -np 4 ./main ../graph_datasets/dblp.edgelist $k >> cliques.results
done





