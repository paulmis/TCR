#include <bits/stdc++.h>
using namespace std;

#define ll              long long
#define all(x)          x.begin(), x.end()
#define nl              cout << "\n";
#define pv(vec)         for(int i = 0; i < vec.size(); i++) cout << vec[i] << " "; nl
#define vb              vector<bool>
#define vi              vector<int>
#define vvi             vector<vector<int>>
#define vii             vector<pair<int, int>>
#define vll             vector<ll>
#define vvll            vector<vector<ll>>
#define pii             pair<int, int>
#define vpii            vector<pii>
#define pb              push_back
#define mp              make_pair
#define fi              first
#define se              second
#define pi              acos(-1)
#define eps             1e-8
#define _(x)            {cout << #x << " = " << x << "\n";}

// Binsearch
int l = 0, r = arr.size() - 1;
while (l <= r) {
    int m = l + (r - l) / 2;
    if (table[m] == key) return true;
    if (table[m] > key)  r = m - 1;
    else                 l = m + 1;
}
return false;
// ---------------------------- //

// Bionominal FOR MODULO USE INV FACTORIAL
int C(int n, int k) {
    double res = 1;
    for (int i = 1; i <= k; ++i)
        res = res * (n - k + i) / i;
    return (int)(res + 0.01);
}

int C[maxn + 1][maxn + 1];
void pascal() {
    C[0][0] = 1;
    for (int n = 1; n <= maxn; ++n) {
        C[n][0] = C[n][n] = 1;
        for (int k = 1; k < n; ++k)
            C[n][k] = C[n - 1][k - 1] + C[n - 1][k];
    }
}

(n choose k) = (n choose (n - k))
(n choose k) = n/k((n - 1) choose (k - 1))
sum_(k->n)(n choose k) = 2^n
sum_(m->n)(m choose k) = ((n + 1) choose (k + 1))
sum_(k->m)((n + k) choose k) = ((n + m + 1) choose m)
sum_(k->n)((n choose k)^2) = (2n choose n)
sum_(k=1->n)(k * (n choose k)) = n2^(n-1)
sum_(k->n)((n - k) choose k) = fib_(n+1)
// ---------------------------- //

// Catalan numbers
vll cat;
void catalan() {
    cat[0] = cat[1] = 1;
    for (int i = 2; i <= n; i++) {
        cat[i] = 0;
        for (int j = 0; j < i; j++) {
            cat[i] += (cat[j] * cat[i - j - 1]) % mod;
            if (cat[i] >= mod)
                cat[i] -= mod;
        }
    }
}

1, 1, 2, 5, 14, 42, 132, 429, 1430, 4862, 16796, 58786, 208012, 742900, ...
Cn = 1/(n + 1)(2n choose n)

Number of correct bracket sequence consisting of n openingand n closing brackets.
The number of rooted full binary trees with n + 1 leaves(vertices are not numbered).A rooted binary tree is full if every vertex has either two children or no children.
The number of ways to completely parenthesize n + 1 factors.
The number of triangulations of a convex polygon with n + 2 sides(i.e.the number of partitions of polygon into disjoint triangles by using the diagonals).
The number of ways to connect the 2n points on a circle to form n disjoint chords.
The number of non - isomorphic full binary trees with n internal nodes(i.e.nodes having at least one son).
The number of monotonic lattice paths from point(0, 0) to point(n, n) in a square lattice of size n×n, which do not pass above the main diagonal(i.e.connecting(0, 0) to(n, n)).
Number of permutations of length n that can be stack sorted(i.e.it can be shown that the rearrangement is stack sorted ifand only if there is no such index i < j < k, such that ak < ai < aj).
The number of non-crossing partitions of a set of n elements.
The number of ways to cover the ladder 1…n using n rectangles(The ladder consists of n columns, where ith column has a height i).
// ---------------------------- //

// FFT [polynomial multiplication] O(nlogn) 
void fft(vector<cd>& a, bool invert) {
    int n = a.size();
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) swap(a[i], a[j]);
    }

    for (int len = 2; len <= n; len <<= 1) {
        double ang = 2 * pi / len * (invert ? -1 : 1);
        cd wlen(cos(ang), sin(ang));
        for (int i = 0; i < n; i += len) {
            cd w(1);
            for (int j = 0; j < len / 2; j++) {
                cd u = a[i + j], v = a[i + j + len / 2] * w;
                a[i + j] = u + v;
                a[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }

    if (invert)
        for (cd& x : a)
            x /= n;
}
// ---------------------------- //

// N-th fibonnaci number O(logn)
int fib(int n) {
    if (n == 0) return 0;
    if (n < 3) return (F[n] = 1);
    if (!F[n]) {
        int k = (n & 1) ? (n + 1) / 2 : n / 2;
        F[n] = (n & 1) ? (fib(k) * fib(k) + fib(k - 1) * fib(k - 1))
            : (2 * fib(k - 1) + fib(k)) * fib(k);
    }
    return F[n];
}
// ---------------------------- //

ll gcd(ll a, ll b) {
    return b == 0 ? a : gcd(b, a % b);
}

// Assumes a and b aren't 0
ll lcm(ll a, ll b) {
    return a / _gcd * b;
}

ll gcdExt(ll a, ll b, ll& x, ll& y) {
    if (b == 0) {
        x = 1;
        y = 0;
        return a;
    }
    ll x1, y1;
    ll d = gcdExt(b, a % b, x1, y1);
    x = y1;
    y = x1 - y1 * (a / b);
    return d;
}

// ax0+by0=g, g id gcd
// if sol. exists, all numbers in form x=x0+kb/g, y=y0-ka/g are solutions
bool diophantineExists(ll a, ll b, ll c, ll& x0, ll& y0, ll& g) {
    g = gcdExt(abs(a), abs(b), x0, y0);
    if (c % g) return false;
    x0 *= c / g;
    y0 *= c / g;
    if (a < 0) x0 = -x0;
    if (b < 0) y0 = -y0;
    return true;
}
// ---------------------------- //

// # of distinct ways to sum integers to n
ll intPart(int n) {
	if (n < 2) return P[n] = 1;
	if (!P[n])
		for (int k = 1; (k * (3 * k - 1)) / 2 <= n; k++)
			for (int i = 1; i < 4 && k * (3 * k + i - 2) / 2 <= n; i += 2)
				P[n] += ((k & 1 ? 1 : -1) * intPart(n - (k * (3 * k + i - 2) / 2)));
	return P[n];
}
// ---------------------------- //

// Inverse factorials
fac[0] = ifac[0] = ifac[1] = 1;
for (int i = 1; i < maxn; i++)
    fac[i] = fac[i - 1] * i % mod;
ifac[maxn - 1] = mpow(fac[maxn - 1], mod - 2);
for (int i = maxn - 2; i > 1; i--)
    ifac[i] = ifac[i + 1] * (i + 1) % mod;
// ---------------------------- //

// Find last remaining number when crossing out every k-th number O(n)
int josephus(int n, int k) {
    int res = 0;
    for (int i = 1; i <= n; ++i)
        res = (res + k) % i;
    return res + 1;
}

// O(klogn)
int josephus(int n, int k) {
    if (n == 1) return 0;
    if (k == 1) return n - 1;
    if (k > n)  return (josephus(n - 1, k) + k) % n;
    int cnt = n / k;
    int res = josephus(n - cnt, k);
    res -= n % k;
    return (res < 0 ? res + n : res + res / (k - 1));
}

Analytical for k=2
J_n,2 = 1 + 2(n - 2^(floor(log(2, n))))
// ---------------------------- //

// Matrix determinant O(n^3)
double det(vector<vector<double>>& a) {
    int n = a.size();
    double det = 1;
    for (int i = 0; i < n; i++) {
        int k = i;
        for (int j = i + 1; j < n; j++)
            if (abs(a[j][i]) > abs(a[k][i]))
                k = j;
        if (abs(a[k][i]) < EPS) {
            det = 0;
            break;
        }
        swap(a[i], a[k]);
        if (i != k)
            det = -det;
        det *= a[i][i];
        for (int j = i + 1; j < n; ++j)
            a[i][j] /= a[i][i];
        for (int j = 0; j < n; ++j)
            if (j != i && abs(a[j][i]) > EPS)
                for (int k = i + 1; k < n; ++k)
                    a[j][k] -= a[i][k] * a[j][i];
    }
    return det;
}
// ---------------------------- //

using u64 = uint64_t;
using u128 = __uint128_t;

bool check_composite(u64 n, u64 a, u64 d, int s) {
    u64 x = mpow(a, d, n);
    if (x == 1 || x == n - 1) return false;
    for (int r = 1; r < s; r++) {
        x = (u128)x * x % n;
        if (x == n - 1) return false;
    }
    return true;
};

// Miller-Rabin [check if n is prime] 
// O(k*(logn)^3), k - size of a
bool MillerRabin(u64 n) {
    if (n < 2) return false;
    int r = 0; u64 d = n - 1;
    while ((d & 1) == 0) {
        d >>= 1;
        r++;
    }
    for (int a : {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37}) {
        if (n == a) return true;
        if (check_composite(n, a, d, r))  return false;
    }
    return true;
}
// ---------------------------- //

// Modual exponentation
ll mpow(ll base, ll exp, ll mod) {
    ll res = 1;
    while (exp) {
        if (exp & 1) res = (res * base) % mod;
        base = (base * base) % mod;
        exp >>= 1;
    }
    return res;
}
// ---------------------------- //

vb isprime;
// Erasthotenes Sieve O(nloglogn)
void sieve(n) {
    isprime.assign(n + 1, true);
    isprime[0] = isprime[1] = false;
    for (int i = 2; i <= n; i++)
        if (isprime[i] && (ll)i * i <= n)
            for (int j = i * i; j <= n; j += i)
                isprime[j] = false;
}
// ---------------------------- //

vi ans;
vvi combinations;

void gen(int n, int k, int idx, bool rev) {
	if (k > n || k < 0) return;
	if (!n) {
		for (int i = 0; i < idx; ++i) {
			combinations.push_back(vi());
			if (ans[i])
				combinations[combinations.size()].pb(i + 1);
		}
		return;
	}
	ans[idx] = rev;
	gen(n - 1, k - rev, idx + 1, false);
	ans[idx] = !rev;
	gen(n - 1, k - !rev, idx + 1, true);
}

// Steinhaus–Johnson–Trotter algorithm [generate all combinations by swapping one element each iteration] O(n * (n choose k))
void adjecentSwapCombinations(int n, int k) {
	ans.resize(n);
	gen(n, k, 0, false);
}
// ---------------------------- //

// Totient function [# of numbers rel. prime to n smaller in [1, n)] O(sqrt(n))
int phi(int n) {
    int result = n;
    for (int i = 2; i * i <= n; i++) {
        if (n % i == 0) {
            while (n % i == 0)
                n /= i;
            result -= result / i;
        }
    }
    if (n > 1)
        result -= result / n;
    return result;
}

// O(nloglogn)
vi phi1ToN(int n) {
    vi phi(n + 1);
    phi[0] = 0; phi[1] = 1;
    for (int i = 2; i <= n; i++) phi[i] = i;
    for (int i = 2; i <= n; i++)
        if (phi[i] == i)
            for (int j = i; j <= n; j += i)
                phi[j] -= phi[j] / i;
    return phi;
}

(1) Sum of the totient functions of all positive divisors of n is equal to n
(2) If p is prime then gcd(p, q) = 1 for 1 <= q < p
    Therefore phi(p) = p - 1
(3) If a and b are relatively prime, then phi(ab) = phi(a) * phi(b)
(4) If p is prime and k >= 1 then there are exactly p^k/p numbers between 1 and p^k that are divisible by p
    Therefore phi(p^k) = p^k - p^(k - 1)
(2) If a and m are relatively prime then a^(phi(m)) === 1 (mod m)
    If m is prime then a^(m - 1) === 1 (mod m)
    Therefore a^n === a^(n mod phi(m)) (mod m) -> easy a^n computation for big n
// ---------------------------- //

// 2-SAT
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

// 2-satisfiability O(V + E) 
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
// ---------------------------- //

vvi adj;
vb visited, isap;
vi tin, low;
int timer;

void dfs(int v, int p = -1) {
    visited[v] = true;
    tin[v] = low[v] = timer++;
    int children = 0;
    for (int to : adj[v]) {
        if (to == p) continue;
        if (visited[to])
            low[v] = min(low[v], tin[to]);
        else {
            dfs(to, v);
            low[v] = min(low[v], low[to]);
            if (low[to] >= tin[v] && p != -1)
                isap[v] = true;
            ++children;
        }
    }
    if (p == -1 && children > 1)
        isap[v] = true;
}

// Articulation points [verticies that would split the graph into two components upon removal] in O(V + E)
void findArticulationPoints() {
    int n = adj.size();
    timer = 0;
    isap.assign(n, false); visited.assign(n, false);
    tin.assign(n, -1); low.assign(n, -1);
    for (int i = 0; i < n; i++)
        if (!visited[i])
            dfs(i);
}
// ---------------------------- //

// Bellman Ford [find any negative cycle] O(V^2)
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
// ---------------------------- //

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
// ---------------------------- //

// Bridges
vb visited;
vpii bridges;
vi tin, low;
int timer;

void dfs(int v, int p = -1) {
	visited[v] = true;
	tin[v] = low[v] = timer++;
	for (int to : adj[v]) {
		if (to == p) continue;
		if (visited[to])
			low[v] = min(low[v], tin[to]);
		else {
			dfs(to, v);
			low[v] = min(low[v], low[to]);
			if (low[to] > tin[v])
				bridges.push_back({ v, to });
		}
	}
}

// Find all bridges [edges that would split the graph into two components upon removal] O(V + E)
void findBridges() {
	int n = adj.size();
	timer = 0;
	visited.assign(n, false); bridges.clear();
	tin.assign(n, -1); low.assign(n, -1);
	for (int i = 0; i < n; ++i)
		if (!visited[i])
			dfs(i);
}
// ---------------------------- //

struct node {
    int idx;
    ll dist;
}; bool operator<(const node& l, const node& r) { return l.dist < r.dist; }

// Dijkstra O(n^2 + m)
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
// ---------------------------- //

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

// Find the maximum flow of a directed graph O(E*sqrt(V))
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
// ---------------------------- //

struct node {
    int p, size = 1;
}; vector<node> ds;

void build(int size) {
    ds.resize(size);
    for (int i = 0; i < size; i++) ds[i].p = i;
}

// Check connectivity O(a(n))
int find(int u) {
    if (u != ds[u].p) ds[u].p = find(ds[u].p);
    return ds[u].p;
}

// Merge components O(a(n))
void merge(int u, int v) {
    if ((u = find(u)) != (v = find(v))) {
        if (ds[u].size < ds[v].size) std::swap(u, v);
        ds[v].p = ds[u].p;
        ds[u].size += ds[v].size;
    }
}
// ---------------------------- //

// Eulerian path [passes through all edges exactly once] O(m)
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
// ---------------------------- //

// All shortest paths in O(V^3)
// Faster than dijkstra for dense graphs
vll FloydWarshall(){
	int n = adj.size();
	vvll dist(n, vll(n, LLONG_MAX));
	for (int u = 0; u < V; u++) {
		dist[u][u] = 0;
		for (edge e : adj[u]) dist[u][e.fi] = e.se;
	}
	for (int i = 0; i < n; i++)
		for (int u = 0; u < n; u++)
			for (int v = 0; v < n; v++)
				if (dist[u][i] != LLONG_MAX && dist[i][v] != LLONG_MAX)
					dist[u][v] = min(dist[u][v], dist[u][i] + dist[i][v]);
	return dist;
}
// ---------------------------- //

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

// Find SCCs O(V + E)
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
// ---------------------------- //

// Lowest common ancestor (with binary lifting)
int timer, n, l;
vi tin, tout;
vvi up;

void dfs(int v, int p) {
    tin[v] = ++timer;
    up[v][0] = p;
    for (int i = 1; i <= l; ++i)
        up[v][i] = up[up[v][i - 1]][i - 1];
    for (int u : adj[v])
        if (u != p)
            dfs(u, v);
    tout[v] = ++timer;
}

bool isancestor(int u, int v) {
    return tin[u] <= tin[v] && tout[u] >= tout[v];
}

// O(logn)
int lca(int u, int v) {
    if (isancestor(u, v)) return u;
    if (isancestor(v, u)) return v;
    for (int i = l; i >= 0; --i)
        if (!isancestor(up[u][i], v))
            u = up[u][i];
    return up[u][0];
}

// O(nlogn)
void preprocess(int root) {
    tin.resize(n); tout.resize(n);
    timer = 0;
    l = ceil(log2(n));
    up.assign(n, vi(l + 1));
    dfs(root, root);
}
// ---------------------------- //

MST Count
Let A be the adjacency matrix of the graph : Au, v is the number of edges between uand v.Let D be the degree matrix of the graph : a diagonal matrix with Du, u being the degree of vertex u(including multiple edgesand loops - edges which connect vertex u with itself).

The Laplacian matrix of the graph is defined as L = D - A.According to Kirchhoff's theorem, all cofactors of this matrix are equal to each other, and they are equal to the number of spanning trees of the graph. The (i,j) cofactor of a matrix is the product of (-1)i+j with the determinant of the matrix that you get after removing the i-th row and j-th column. So you can, for example, delete the last row and last column of the matrix L, and the absolute value of the determinant of the resulting matrix will give you the number of spanning trees.
// ---------------------------- //

// Segment trees (dynamic range queries and updates)
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

// LAZY (ranged updates, reuse above definitions)
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
// ---------------------------- //

inline int log2Up(int n) {
    int res = 0;
    while ((1 << res) < n) res++;
    return res;
}

// Define associative operation (x * y) * z = x * (y * z)
SqrtTreeItem op(const SqrtTreeItem& a, const SqrtTreeItem& b);

// Answer range queries on associative operators
// Build: O(nloglogn), query: O(1), update: O(sqrt(n))
class SqrtTree {
private:
    int n, lg, indexSz;
    vector<SqrtTreeItem> v;
    vector<int> clz, layers, onLayer;
    vector< vector<SqrtTreeItem> > pref, suf, between;

    inline void buildBlock(int layer, int l, int r) {
        pref[layer][l] = v[l];
        for (int i = l + 1; i < r; i++) pref[layer][i] = op(pref[layer][i - 1], v[i]);
        suf[layer][r - 1] = v[r - 1];
        for (int i = r - 2; i >= l; i--) suf[layer][i] = op(v[i], suf[layer][i + 1]);
    }

    inline void buildBetween(int layer, int lBound, int rBound, int betweenOffs) {
        int bSzLog = (layers[layer] + 1) >> 1;
        int bCntLog = layers[layer] >> 1;
        int bSz = 1 << bSzLog;
        int bCnt = (rBound - lBound + bSz - 1) >> bSzLog;
        for (int i = 0; i < bCnt; i++) {
            SqrtTreeItem ans;
            for (int j = i; j < bCnt; j++) {
                SqrtTreeItem add = suf[layer][lBound + (j << bSzLog)];
                ans = (i == j) ? add : op(ans, add);
                between[layer - 1][betweenOffs + lBound + (i << bCntLog) + j] = ans;
            }
        }
    }

    inline void buildBetweenZero() {
        int bSzLog = (lg + 1) >> 1;
        for (int i = 0; i < indexSz; i++) v[n + i] = suf[0][i << bSzLog];
        build(1, n, n + indexSz, (1 << lg) - n);
    }

    inline void updateBetweenZero(int bid) {
        int bSzLog = (lg + 1) >> 1;
        v[n + bid] = suf[0][bid << bSzLog];
        update(1, n, n + indexSz, (1 << lg) - n, n + bid);
    }

    void build(int layer, int lBound, int rBound, int betweenOffs) {
        if (layer >= (int)layers.size()) return;
        int bSz = 1 << ((layers[layer] + 1) >> 1);
        for (int l = lBound; l < rBound; l += bSz) {
            int r = min(l + bSz, rBound);
            buildBlock(layer, l, r);
            build(layer + 1, l, r, betweenOffs);
        }
        if (layer == 0) buildBetweenZero();
        else            buildBetween(layer, lBound, rBound, betweenOffs);
    }

    void update(int layer, int lBound, int rBound, int betweenOffs, int x) {
        if (layer >= (int)layers.size()) return;
        int bSzLog = (layers[layer] + 1) >> 1;
        int bSz = 1 << bSzLog;
        int blockIdx = (x - lBound) >> bSzLog;
        int l = lBound + (blockIdx << bSzLog);
        int r = min(l + bSz, rBound);
        buildBlock(layer, l, r);
        if (layer == 0) updateBetweenZero(blockIdx);
        else            buildBetween(layer, lBound, rBound, betweenOffs);
        update(layer + 1, l, r, betweenOffs, x);
    }

    inline SqrtTreeItem query(int l, int r, int betweenOffs, int base) {
        if (l == r)     return v[l];
        if (l + 1 == r) return op(v[l], v[r]);
        int layer = onLayer[clz[(l - base) ^ (r - base)]];
        int bSzLog = (layers[layer] + 1) >> 1;
        int bCntLog = layers[layer] >> 1;
        int lBound = (((l - base) >> layers[layer]) << layers[layer]) + base;
        int lBlock = ((l - lBound) >> bSzLog) + 1;
        int rBlock = ((r - lBound) >> bSzLog) - 1;
        SqrtTreeItem ans = suf[layer][l];
        if (lBlock <= rBlock) {
            SqrtTreeItem add = (layer == 0) ? (
                query(n + lBlock, n + rBlock, (1 << lg) - n, n)
                ) : (
                    between[layer - 1][betweenOffs + lBound + (lBlock << bCntLog) + rBlock]
                    );
            ans = op(ans, add);
        }
        ans = op(ans, pref[layer][r]);
        return ans;
    }
public:
    inline SqrtTreeItem query(int l, int r) {
        return query(l, r, 0, 0);
    }

    inline void update(int x, const SqrtTreeItem& item) {
        v[x] = item;
        update(0, 0, n, 0, x);
    }

    SqrtTree(const vector<SqrtTreeItem>& a)
        : n((int)a.size()), lg(log2Up(n)), v(a), clz(1 << lg), onLayer(lg + 1) {
        clz[0] = 0;
        for (int i = 1; i < (int)clz.size(); i++) clz[i] = clz[i >> 1] + 1;
        int tlg = lg;
        while (tlg > 1) {
            onLayer[tlg] = (int)layers.size();
            layers.push_back(tlg);
            tlg = (tlg + 1) >> 1;
        }
        for (int i = lg - 1; i >= 0; i--) onLayer[i] = max(onLayer[i], onLayer[i + 1]);
        int betweenLayers = max(0, (int)layers.size() - 1);
        int bSzLog = (lg + 1) >> 1;
        int bSz = 1 << bSzLog;
        indexSz = (n + bSz - 1) >> bSzLog;
        v.resize(n + indexSz);
        pref.assign(layers.size(), vector<SqrtTreeItem>(n + indexSz));
        suf.assign(layers.size(), vector<SqrtTreeItem>(n + indexSz));
        between.assign(betweenLayers, vector<SqrtTreeItem>((1 << lg) + bSz));
        build(0, 0, n, 0);
    }
};
// ---------------------------- //

// Topological sort O(V + E)
vi topsort()
{
	int n = adj.size();
	vector<int> deg(n, 0), tsort;
	for (int u = 0; u < adj.size(); u++)
		for (int v : adj[u])
			deg[v]++;
	queue<int> q;
	for (int u = 0; u < adj.size(); u++)
		if (deg[u] == 0)
			q.push(u);
	while (!q.empty()) {
		int u = q.front(); q.pop();
		tsort.push_back(u);
		for (int v : adj[u])
			if (--deg[v] == 0)
				q.push(v);
	}
	return tsort;
}
// ---------------------------- //

struct trie_node {
    bool is_end; map<char, trie_node*> nodes;
    trie_node() { is_end = false; }
};

class trie {
    bool remove_leaf(trie_node* node) {
        if (!node->is_end && node->nodes.empty()) {
            delete node;
            return true;
        }
        return false;
    }

    bool remove_rec(trie_node* node, string& key, int depth) {
        if (depth == key.size()) {
            node->is_end = false;
            return remove_leaf(node);
        }
        auto search = node->nodes.find(key[depth]);
        if (search == node->nodes.end() ||
            !remove_rec(search->second, key, depth + 1))
            return false;
        node->nodes.erase(search);
        return remove_leaf(node);
    }
    trie_node* root;

public:
    trie()
        : root(new trie_node()) {}
    void insert(string key) {
        trie_node* node = root;
        for (char c : key) {
            if (node->nodes.find(c) == node->nodes.end())
                node->nodes[c] = new trie_node();
            node = node->nodes[c];
        }
        node->is_end = true;
    }
    bool search(string key) {
        trie_node* node = root;
        for (char c : key) {
            if (node->nodes.find(c) == node->nodes.end())
                return false;
            node = node->nodes[c];
        }
        return node->is_end;
    }
    inline void remove(string key) { remove_rec(root, key, 0); }
};
// ---------------------------- //

struct pt {
    int x, y, id;
};
double mindist;
pii best;
vector<pt> a;

struct cmp_x {
    bool operator()(const pt& a, const pt& b) const { return a.x < b.x || (a.x == b.x && a.y < b.y); }
};
struct cmp_y {
    bool operator()(const pt& a, const pt& b) const { return a.y < b.y; }
};

void upd_ans(const pt& a, const pt& b) {
    double dist = sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
    if (dist < mindist) {
        mindist = dist;
        best = { a.id, b.id };
    }
}

vector<pt> t;
void nearestPointRec(int l, int r) {
    if (r - l <= 3) {
        for (int i = l; i < r; ++i)
            for (int j = i + 1; j < r; ++j)
                upd_ans(a[i], a[j]);
        sort(a.begin() + l, a.begin() + r, cmp_y());
        return;
    }

    int m = (l + r) >> 1;
    int midx = a[m].x;
    nearestPointRec(l, m);
    nearestPointRec(m, r);
    merge(a.begin() + l, a.begin() + m, a.begin() + m, a.begin() + r, t.begin(), cmp_y());
    copy(t.begin(), t.begin() + r - l, a.begin() + l);

    int tsz = 0;
    for (int i = l; i < r; ++i) {
        if (abs(a[i].x - midx) < mindist) {
            for (int j = tsz - 1; j >= 0 && a[i].y - t[j].y < mindist; --j)
                upd_ans(a[i], t[j]);
            t[tsz++] = a[i];
        }
    }
}

// Find nearest point O(nlogn)
pii nearestPoint() {
    t.assign(a.size(), pt());
    sort(all(a), cmp_x());
    mindist = 1E20;
    nearestPointRec(0, a.size());
    return best;
}
// ---------------------------- //

Picks theorem
S = I + B/2 - 1
... given a polygon is a lattice polygon
S - surface area of the polygon
I - number of lattice points inside of the polygon
B - number of lattice points lying on the sides of the polygon
for v1, v2: B = gcd(|v1.x-v2.x|, |v1.y-v2.y|) - 1 

// Polygon area
double area(vector<point>& polygon) {
    double res = 0;
    for (i = 0; i < polygon.size(); i++) {
        point p = i ? fig[i - 1] : fig.back();
        point q = fig[i];
        res += (p.x - q.x) * (p.y + q.y);
    }
    return fabs(res) / 2;
}

Triangle clockwise/counterclockwise orientation: sign of cross product

struct pt {
    double x, y;
};

int orientation(pt a, pt b, pt c) {
    double v = a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y);
    if (v < 0) return -1; // clockwise
    if (v > 0) return +1; // counter-clockwise
    return 0;
}

bool cw(pt a, pt b, pt c, bool inclColinear) {
    int o = orientation(a, b, c);
    return o < 0 || (inclColinear && o == 0);
}
bool collinear(pt a, pt b, pt c) { return orientation(a, b, c) == 0; }

// Graham scan [Min. bounding polygon] O(nlogn)
vector<pt> convexHull(vector<pt>& a, bool inclColinear = false) {
    pt p0 = *min_element(all(a), [](pt a, pt b) {
        return tie(a.y, a.x) < tie(b.y, b.x); // or make_pair
        });
    sort(all(a), [&p0](const pt& a, const pt& b) {
        int o = orientation(p0, a, b);
        if (o == 0)
            return (p0.x - a.x) * (p0.x - a.x) + (p0.y - a.y) * (p0.y - a.y)
            < (p0.x - b.x) * (p0.x - b.x) + (p0.y - b.y) * (p0.y - b.y);
        return o < 0;
        });
    if (inclColinear) {
        int i = (int)a.size() - 1;
        while (i >= 0 && collinear(p0, a[i], a.back())) i--;
        reverse(a.begin() + i + 1, a.end());
    }

    vector<pt> st;
    for (int i = 0; i < (int)a.size(); i++) {
        while (st.size() > 1 && !cw(st[st.size() - 2], st.back(), a[i], inclColinear))
            st.pop_back();
        st.push_back(a[i]);
    }
    return st;
}

// Rotating calipers [max distance] O(n)
pt maxDist(vector<pt>& poly) {
    int n = poly.size();
    pt res(0.0, 0.0);
    for (int i = 0, j = n < 2 ? 0 : 1; i < j; i++)
        for (;; j = next(j, n)) {
            res = max(res, dist2(poly[i], poly[j]));
            if (ccw(poly[i + 1] - poly[i], poly[next(j, n)] - poly[j]) >= 0) break;
        }
    return res;
}
// ---------------------------- //

// Pi function O(n)
vi computepi(string& s) {
	int n = s.size();
	vi p(n);
	for (int i = 1; i < n; i++) {
		int j = p[i - 1];
		while (j > 0 && s[i] != s[j]) j = p[j - 1];
		if (s[i] == s[j]) j++;
		p[i] = j;
	}
	return p;
}

The prefix function for this string is defined as an array p of length n, where p[i] is the length of the longest proper prefix of the substring s[0..i] which is also a suffix of this substring.A proper prefix of a string is a prefix that is not equal to the string itself.By definition, p[0] = 0.
Counting the number of occurrences of each prefix
	vi ans(n + 1);
	for (int i = 0; i < n; i++)		ans[p[i]]++;
	for (int i = n - 1; i > 0; i--) ans[p[i - 1]] += ans[i];
	for (int i = 0; i <= n; i++)	ans[i]++;

// Knuth-Morris-Pratt [Find s in t] O(t)
int kmp(string& s, string& t) {
	vi p = computepi(s);
	for (int i = 0, j = 0; i < t.size(); i++) {
		while (j > 0 && t[i] != s[j]) j = p[j - 1];
		if (t[i] == s[j] && ++j == s.size()) return i - j + 1;
	}
	return -1;
}
// ---------------------------- //

// Longest common subsequence O(n^2)
string LCseq(string& A, string& B) {
    string str;
    int a = A.size(), b = B.size();
    vvi DP(a + 1, vi(b + 1, 0));
    for (int i = 1; i <= a; i++)
        for (int j = 1; j <= b; j++)
        {
            if (A[i - 1] == B[j - 1]) DP[i][j] = DP[i - 1][j - 1] + 1;
            else                      DP[i][j] = max(DP[i - 1][j], DP[i][j - 1]);
        }
    for (int i = a - 1, j = b - 1; i >= 0 && j >= 0; ) {
        if (A[i] == B[j]) {
            str += A[i];
            i--, j--;
        } else {
            if (DP[i + 1][j] < DP[i][j + 1]) i--;
            else                             j--;
        }
    }
    reverse(all(str));
    return str;
}
// ---------------------------- //

// Levenshtein distance [min. edits to transform A to B] O(n^2)
int levenshtein(string& A, string& B)
{
	int a = A.size(), b = B.size();
	vvi d(a, vi(b, 0));
	for (int i = 1; i <= a; ++i) d[i][0] = i;
	for (int i = 1; i <= b; ++i) d[0][i] = i;
	for (int i = 1; i <= a; ++i)
		for (int j = 1; j <= b; ++j)
			d[i][j] = std::min({ d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (A[i - 1] == B[j - 1] ? 0 : 1) });
	return d[a][b];
}
// ---------------------------- //

// Longest increasing subsequence O(nlogn)
int lis(vi& arr) {
	int n = arr.size();
	vi d(n + 1, INT_MAX);
	d[0] = -INT_MAX;
	for (int i = 0; i < n; i++) {
		int j = upper_bound(d.begin(), d.end(), arr[i]) - d.begin();
		if (d[j - 1] < arr[i] && arr[i] < d[j]) d[j] = arr[i];
	}

	int ans = 0;
	for (int i = 0; i <= n; i++)
		if (d[i] < INT_MAX)
			ans = i;
	return ans;
}


It is also possible to restore the subsequence using this approach.This time we have to maintain two auxiliary arrays.One that tells us the index of the elements in d[].And again we have to create an array of "ancestors" p[i].p[i] will be the index of the previous element for the optimal subsequence ending in element i.
It is easy to maintain these two arrays in the course of iteration over the array a[] alongside the computations of d[]. And at the end it is not difficult to restore the desired subsequence using these arrays.
// ---------------------------- //

// Manacher [all sub-palindromes] O(n)
// d1, d2 - number of palindromes with odd/even lengths with centers in i
pair<vi, vi> manacher(vi& arr) {
    int n = arr.size();
    vector<int> d1(n), d2(n);
    for (int i = 0, l = 0, r = -1; i < n; i++) {
        int k = (i > r) ? 1 : min(d1[l + r - i], r - i + 1);
        while (0 <= i - k && i + k < n && s[i - k] == s[i + k]) k++;
        d1[i] = k--;
        if (i + k > r) {
            l = i - k;
            r = i + k;
        }
    }
    for (int i = 0, l = 0, r = -1; i < n; i++) {
        int k = (i > r) ? 0 : min(d2[l + r - i + 1], r - i + 1);
        while (0 <= i - k - 1 && i + k < n && s[i - k - 1] == s[i + k]) k++;
        d2[i] = k--;
        if (i + k > r) {
            l = i - k - 1;
            r = i + k;
        }
    }
    return { d1, d2 };
}
// ---------------------------- //

// Rabin-Karp [Determine all occurances of s in t] O(s + t)
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
// ---------------------------- //

// Sparse table (static range queries)
int ts, k;
vvll st;
vi lt;

// O(nlogn) build
void build(vll& arr) {
	k = 1; ts = arr.size();
	lt.assign(arr.size(), 0);
	for (int i = 2; i <= ts; i++)
		lt[i] = lt[i / 2] + 1;
	for (int p = 1; p < arr.size(); p <<= 1, k++);
	st.resize(ts + 1, vll(k + 1, 0));
	for (int i = 0; i < ts; i++)
		st[i][0] = arr[i];
	for (int j = 1; j <= k; j++)
		for (int i = 0; i + (1 << j) <= ts; i++)
			st[i][j] = min(st[i][j - 1], st[i + (1 << (j - 1))][j - 1]);
}

// O(1) RMQ
ll min(int l, int r) { // inclusive
	int j = lt[r - l + 1];
	int minimum = min(st[l][j], st[r - (1 << j) + 1][j]);
}