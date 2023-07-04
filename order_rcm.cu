// See header for general documentation

#include <algorithm>
#include "order_rcm.cuh"
#include "tools.cuh"
#include "edgelist.cuh"
#include "adjlist.cuh"
#include "order_deg.cuh"
#include "algo_bfs.cuh"


using namespace std;


vector<ul> order_rcm(Edgelist &h, bool directed) {
  vector<ul> rank_deg = order_deg(h, false); // degree ASC
  vector<ul> rank_bfs;

  if(directed) {
    Dadjlist g(h, rank_deg);
    rank_bfs = rank_from_order(algo_bfs(g)); // Cuthill–McKee
  }
  else {
    Uadjlist g(h, rank_deg);
    rank_bfs = rank_from_order(algo_bfs(g)); // Cuthill–McKee
  }

  vector<ul> rank(h.n); //rank.reserve(h.n);
  for (ul u = 0; u < h.n; u++) {
    rank[u] = h.n-1 - rank_bfs[rank_deg[u]]; // compose reverse Cuthill–McKee with degree
  }
  return rank;
}
