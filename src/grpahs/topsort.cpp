vi topsort()
{
	int n = adj.size();
	vector<int> deg(n, 0), tsort;
	for (int u = 0; u < adj.size(); u++)
		for (int v : adj[u])
			deg[v]++;
	queue<int> q;
	for (int u = 0; u < adj.size(); u++)
		if (deg[u] == 0)
			q.push(u);
	while (!q.empty()) {
		int u = q.front(); q.pop();
		tsort.push_back(u);
		for (int v : adj[u])
			if (--deg[v] == 0)
				q.push(v);
	}
	return tsort;
}