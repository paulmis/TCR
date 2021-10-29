struct node {
    int p, size = 1;
}; vector<node> ds;

void build(int size) {
    ds.resize(size);
    for (int i = 0; i < size; i++) ds[i].p = i;
}

int find(int u) {
    if (u != ds[u].p) ds[u].p = find(ds[u].p);
    return ds[u].p;
}

void merge(int u, int v) {
    if ((u = find(u)) != (v = find(v))) {
        if (ds[u].size < ds[v].size) std::swap(u, v);
        ds[v].p = ds[u].p;
        ds[u].size += ds[v].size;
    }
}