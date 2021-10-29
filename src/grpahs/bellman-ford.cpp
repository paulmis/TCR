vi BellmanFord(int src, int n)
{
    vi p(edges.size() - 1);
    vll dist(n, LLONG_MAX);
    dist[src] = 0;

    int x;
    for (int i = 0; i < edges.size(); i++) {
        x = -1;
        for (int j = 0; j < edges.size(); ++j)
            if (dist[edges[j].from] < LLONG_MAX)
                if (dist[edges[j].to] > dist[edges[j].from] + edges[j].weight) {
                    dist[edges[j].to] = max(-LLONG_MAX, dist[edges[j].from] + edges[j].weight);
                    p[edges[j].to] = edges[j].from;
                    x = edges[j].to;
                }
    }

    if (x == -1) return vi();
    int y = x;
    for (int i = 0; i < n; ++i)
        y = p[y];

    vi path;
    for (int cur = y;; cur = p[cur]) {
        path.push_back(cur);
        if (cur == y && path.size() > 1)
            break;
    }
    reverse(all(path));

    return path;
}
