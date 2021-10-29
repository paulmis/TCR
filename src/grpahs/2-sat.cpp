vb used, assignment;
vi order, comp;
vvi rev;

void dfs1(int v) {
    used[v] = true;
    for (int u : g[v])
        if (!used[u])
            dfs1(u);
    order.push_back(v);
}

void dfs2(int v, int cl) {
    comp[v] = cl;
    for (int u : rev[v])
        if (comp[u] == -1)
            dfs2(u, cl);
}

bool sat2() {
    int n = adj.size();
    order.clear();
    used.assign(n, false), assignment.assign(n / 2, false);
    for (int i = 0; i < n; i++) {
        for (int j : adj[i]) rev[j].pb(i);
        if (!used[i]) dfs1(i);
    }

    comp.assign(n, -1);
    for (int i = 0, j = 0; i < n; i++) {
        int v = order[n - i - 1];
        if (comp[v] == -1)
            dfs2(v, j++);
    }
    for (int i = 0; i < n; i += 2) {
        if (comp[i] == comp[i + 1]) return false;
        assignment[i / 2] = comp[i] > comp[i + 1];
    }
    return true;
}
