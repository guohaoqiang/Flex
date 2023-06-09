#include "order_deg.cuh"
#include "adjlist.cuh"
#include "edgelist.cuh"
#include <algorithm>

using namespace std;

inline bool compare_nodedeg_desc (Keyvalue &a, Keyvalue &b) { // sort degree DESC, node ASC
	return ( a.val > b.val or (a.val == b.val and a.key < b.key) );
}
inline bool compare_nodedeg_asc (Keyvalue &a, Keyvalue &b) { // sort degree ASC, node ASC
	return ( a.val < b.val or (a.val == b.val and a.key < b.key) );
}


// --------------------------------------------------
// ----------------- From edgelist ------------------
// --------------------------------------------------
vector<ul> rank_from_deg(const vector<ul> &deg, const ul &n, bool desc) {
  vector<Keyvalue> toSort; toSort.reserve(n);
  for (ul u = 0; u < n; ++u){
    //printf("%d of %s: %d\n",__LINE__, __FILE__, deg[u]);
    toSort.push_back( Keyvalue(u, deg[u]) );
  }
  //printf("%d of %s: %d\n",__LINE__, __FILE__, toSort.size());

  if(desc) sort(toSort.begin(), toSort.end(), compare_nodedeg_desc);
  else     sort(toSort.begin(), toSort.end(), compare_nodedeg_asc);

  //printf("%d of %s: %d\n",__LINE__, __FILE__, toSort.size());
  vector<ul> rank(n); //rank.reserve(n);
  for (ul u = 0; u < n; ++u){	
      //printf("%d of %s: [%d]:%d\n",__LINE__, __FILE__, toSort[u].key, u);
      rank[ toSort[u].key ] = u;
  }
  //printf("%d of %s: %d\n",__LINE__, __FILE__, rank.size());

  return rank;
}

vector<ul> order_deg(Edgelist &g, bool desc) {
	Debug("Order deg from edgelist")
  g.compute_degrees();
  return rank_from_deg(g.deg, g.n, desc);
}
vector<ul> order_degOut(Edgelist &g, bool desc) {
	Debug("Order degOut from edgelist")
  g.compute_degrees();
  return rank_from_deg(g.degOut, g.n, desc);
}

vector<ul> order_degIn(Edgelist &g, bool desc) {
	Debug("Order degIn from edgelist")
  g.compute_degrees();
  return rank_from_deg(g.degIn, g.n, desc);
}

// --------------------------------------------------
// ----------------- From adjlist -------------------
// --------------------------------------------------
vector<ul> order_degOut(Adjlist &g) {
	Debug("Order degOut")
    vector<Keyvalue> toSort; toSort.reserve(g.n);
    for (ul u = 0; u < g.n; ++u)
        toSort.push_back( Keyvalue(u, g.get_degree(u)) );

    // cout << "before sort: "<< endl; for (ul i = 0; i < 10; ++i) cout << "\t" << toSort[i].node << " \t " << toSort[i].deg << endl;
    sort(toSort.begin(), toSort.end(), compare_nodedeg_desc);

    vector<ul> rank; rank.reserve(g.n);
    for (ul u = 0; u < g.n; ++u)
    	rank[ toSort[u].key ] = u;

    // cout << "after sort: " << endl; for (ul i = 0; i < 10; ++i) cout << "\t" << toSort[i].node << " \t " << toSort[i].deg << " test:" << rank[i] << endl;
    return rank;
}

vector<ul> order_degIn(Adjlist &g) {
	Debug("Order degIn")
	vector<Keyvalue> toSort; toSort.reserve(g.n);
	for (ul u = 0; u < g.n; ++u)
	    toSort.push_back( Keyvalue(u, 0) );

	for (ul i = 0; i < g.e; ++i)
		toSort[g.adj[i]].val ++;

	sort(toSort.begin(), toSort.end(), compare_nodedeg_desc);

	vector<ul> rank; rank.reserve(g.n);
	for (ul u = 0; u < g.n; ++u)
		rank[ toSort[u].key ] = u;
	return rank;
}
