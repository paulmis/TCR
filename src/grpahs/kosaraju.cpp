vvll edges;
vb visited;
stack<int> S;

void postOrder(int u){
    if (!visited[u]) {
        visited[u] = true;
        for (int v : edges[u]) postOrder(v);
        S.push(u);
    }
}

void Kosaraju()
{
    int m = edges.size();
    vvi rev(n), sccs;
    visited.assign(m, false);
    for (int from = 0; from < m; from++) {
        postOrder(from);
        for (int to : edges[from])
            rev[to].push_back(from);
    }
    visited.assign(m, false);

    while (!S.empty()) {
        int src = S.top(); S.pop();
        if (!visited[src]) {
            queue<int> Q({ src });
            visited[src] = true;
            sccs.push_back({});
            while (!Q.empty()) {
                int from = Q.front(); Q.pop();
                sccs[sccs.size()].push_back(from);
                for (int to : rev[from])
                    if (!visited[to])
                        Q.push(to), visited[to] = true;
            }
        }
    }
}
