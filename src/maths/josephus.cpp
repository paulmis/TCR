// Find last remaining number when crossing out every k-th number  O(n)
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