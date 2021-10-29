// Check if bipartie O(V + E)
bool isBipartie() {
    int n - adj.size();
    vi side(n, -1);
    bool res = true;
    queue<int> q;
    for (int st = 0; st < n; ++st) {
        if (side[st] == -1) {
            q.push(st);
            side[st] = 0;
            while (!q.empty()) {
                int v = q.front();
                q.pop();
                for (int u : adj[v]) {
                    if (side[u] == -1) {
                        side[u] = side[v] ^ 1;
                        q.push(u);
                    }
                    else res &= side[u] != side[v];
                }
            }
        }
    }
    return res;
}
