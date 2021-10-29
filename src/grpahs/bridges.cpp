vb visited;
vpii bridges;
vi tin, low;
int timer;

void dfs(int v, int p = -1) {
	visited[v] = true;
	tin[v] = low[v] = timer++;
	for (int to : adj[v]) {
		if (to == p) continue;
		if (visited[to])
			low[v] = min(low[v], tin[to]);
		else {
			dfs(to, v);
			low[v] = min(low[v], low[to]);
			if (low[to] > tin[v])
				bridges.push_back({ v, to });
		}
	}
}

void findBridges() {
	int n = adj.size();
	timer = 0;
	visited.assign(n, false); bridges.clear();
	tin.assign(n, -1); low.assign(n, -1);
	for (int i = 0; i < n; ++i)
		if (!visited[i])
			dfs(i);
}