#define l(n) (2 * n)
#define r(n) (2 * n + 1)
#define f(x, y) max(x, y) // segfunc
#define def INT_MIN       // default value (e.g. INT_MIN for max(), -1 for gcd)
int ts;                   // tree size (nearest power of 2 larger than arr.size())
vll st;

void build(vll& arr) {
    for (ts = 1; ts < arr.size(); ts <<= 1);
    st.resize(ts * 2, 0);
    for (int i = ts; i < 2 * ts; i++) st[i] = (i - ts < arr.size() ? arr[i - ts] : def);
    for (int i = ts - 1; i > 0; i--)  st[i] = f(st[l(i)], st[r(i)]);
}

void update(int pos, ll val) {
    st[pos += ts] = val;
    while ((pos /= 2) > 0) st[pos] = f(st[l(pos)], st[r(pos)]);
}

ll queryRec(int l, int r, int n, int lhs, int rhs) {
    if (lhs > r || rhs < l)   return def;
    if (lhs >= l && rhs <= r) return st[n];
    int m = lhs + (rhs - lhs) / 2;
    return f((m < l ? def : queryRec(l, r, l(n), lhs, m)),
        (m + 1 > r ? def : queryRec(l, r, r(n), m + 1, rhs)));
}

inline ll query(int l, int r) { return queryRec(l, r, 1, 0, ts - 1); }

// LAZY
vll lazy;

void build(vll& arr) {
    for (ts = 1; ts < arr.size(); ts <<= 1);
    st.resize(ts * 2, 0), lazy.resize(ts * 2, 0);
    for (int i = ts; i < 2 * ts; i++) st[i] = (i - ts < arr.size() ? arr[i - ts] : def);
    for (int i = ts - 1; i > 0; i--) st[i] = f(st[l(i)], st[r(i)]);
}

inline void propagate(int n, int len) {
    if (lazy[n] > 0) {
        st[n] += lazy[n] * len;
        if (len > 1) lazy[l(n)] += lazy[n], lazy[r(n)] += lazy[n];
        lazy[n] = 0;
    }
}

void update(int l, int r, ll val, int n, int lhs, int rhs) {
    if (lhs >= l && rhs <= r) {
        lazy[n] += val;
        propagate(n, rhs - lhs + 1);
    }
    else {
        propagate(n, rhs - lhs + 1);
        int m = lhs + (rhs - lhs) / 2;
        if (m >= l)     update(l, r, val, l(n), lhs, m);
        if (m + 1 <= r) update(l, r, val, r(n), m + 1, rhs);
        st[v] = f(st[l(v)], st[r(v)]);
    }
}

ll query(int l, int r, int n, int lhs, int rhs) {
    propagate(n, rhs - lhs + 1);
    if (lhs >= l && rhs <= r)
        return st[n];
    int m = lhs + (rhs - lhs) / 2;
    return f((m < l ? def : query(l, r, l(n), lhs, m)),
        (m + 1 > r ? def : query(l, r, r(n), m + 1, rhs)));
}
inline ll query(int l, int r) { return query(l, r, 1, 0, ts - 1); }
inline void update(int l, int r, ll val) { update(l, r, val, 1, 0, ts - 1); }