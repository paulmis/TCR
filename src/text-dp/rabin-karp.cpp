// Determine all occurances of s in t in O(s + t)
vi rabin_karp(string& s, string& t) {
    const int p = 31, m = 1e9 + 9;
    int S = s.size(), T = t.size();

    vll pw(max(S, T));
    pw[0] = 1;
    for (int i = 1; i < (int)pw.size(); i++)
        pw[i] = (pw[i - 1] * p) % m;

    vll h(T + 1, 0);
    long long h_s = 0;
    for (int i = 0; i < T; i++) h[i + 1] = (h[i] + (t[i] - 'a' + 1) * pw[i]) % m;
    for (int i = 0; i < S; i++) h_s = (h_s + (s[i] - 'a' + 1) * pw[i]) % m;

    vi occurences;
    for (int i = 0; i + S - 1 < T; i++) {
        long long cur_h = (h[i + S] + m - h[i]) % m;
        if (cur_h == h_s * pw[i] % m)
            occurences.push_back(i);
    }
    return occurences;
}
