// Find a path that passes through all edges exactly once in O(m)
vi eulerianPath() {
    int n = adj.size();
    vector<int> deg(n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            deg[i] += adj[i][j];

    int first = 0;
    while (first < n && !deg[first]) ++first;
    if (first == n) return vi();

    int v1 = -1, v2 = -1;
    bool bad = false;
    for (int i = 0; i < n; ++i)
        if (deg[i] & 1) {
            if (v1 == -1)      v1 = i;
            else if (v2 == -1) v2 = i;
            else               bad = true;
        }
    if (v1 != -1) g[v1][v2]++, g[v2][v1]++;

    stack<int> st; st.push(first);
    vi res;
    while (!st.empty()) {
        int v = st.top(), i;
        for (i = 0; i < n; ++i)
            if (g[v][i])
                break;
        if (i == n) {
            res.push_back(v);
            st.pop();
        }
        else {
            g[v][i]--;
            g[i][v]--;
            st.push(i);
        }
    }

    if (v1 != -1)
        for (int i = 0; i + 1 < res.size(); ++i)
            if ((res[i] == v1 && res[i + 1] == v2) || (res[i] == v2 && res[i + 1] == v1)) {
                vi res2;
                for (int j = i + 1; j < res.size(); ++j) res2.push_back(res[j]);
                for (int j = 1; j <= i; ++j)             res2.push_back(res[j]);
                res = res2;
                break;
            }

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if (g[i][j])
                bad = true;
    return bad ? vi() : res;
}