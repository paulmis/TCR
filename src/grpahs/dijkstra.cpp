struct node {
    int idx;
    ll dist;
}; bool operator<(const node& l, const node& r) { return l.dist < r.dist; }

vll Dijkstra(int src) {
    vll dist;
    priority_queue<node, vector<node>> pq;
    pq.push({ src, dist[src] = 0 });

    while (!pq.empty()) {
        node from = pq.top(); pq.pop();
        for (edge& e : adj[from.idx])
            if (from.dist + e.weight < dist[e.to])
                pq.push({ e.to, dist[e.to] = from.dist + e.weight });
    }
    return dist;
}
