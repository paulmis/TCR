// All shortest paths in O(n^3)
// Faster than dijkstra for dense graphs
vll FloydWarshall() {
	int n = adj.size();
	vvll dist(n, vll(n, LLONG_MAX));
	for (int u = 0; u < V; u++) {
		dist[u][u] = 0;
		for (edge e : adj[u]) dist[u][e.fi] = e.se;
	}
	for (int i = 0; i < n; i++)
		for (int u = 0; u < n; u++)
			for (int v = 0; v < n; v++)
				if (dist[u][i] != LLONG_MAX && dist[i][v] != LLONG_MAX)
					dist[u][v] = min(dist[u][v], dist[u][i] + dist[i][v]);
	return dist;
}