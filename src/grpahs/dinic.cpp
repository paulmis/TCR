vi level, edgeBegin;

bool levelBfs(int s, int t)
{
    fill(all(level), -1);
    level[s] = 0;
    queue<int> q; q.push(s);
    while (!q.empty())
    {
        int from = q.front(); q.pop();
        for (edge& e : edges[from])
            if (level[e.to] < 0 && e.flow < e.capacity) {
                level[e.to] = level[from] + 1;
                Q.push(e.to);
            }
    }
    return level[t] != -1;
}

int pushFlow(int from, int t, int flow)
{
    if (from == t) return flow;
    for (; edgeBegin[from] < edges[from].size(); edgeBegin[from]++) {
        edge& e = edges[from][edgeBegin[from]];
        if (level[e.to] == level[from] + 1 && e.flow < e.capacity) {
            int subpath_flow = pushFlow(e.to, t, min(flow, e.capacity - e.flow));
            if (subpath_flow > 0) {
                e.flow += subpath_flow;
                edges[e.to][e.rev].flow -= subpath_flow;
                return subpath_flow;
            }
        }
    }
    return 0;
}

// Find the maximum flow of a directed graph in O(E*sqrt(V))
// Assumes source can't be the target
int Dinic(int s, int t)
{
    int result = 0;
    // Keep pushing flows until unable to reach the target from the source
    while (levelBfs(s, t)) {
        fill(all(edgeBegin), 0);
        while (int flow = pushFlow(s, t, INT_MAX))
            result += flow;
    }

    return result;
}