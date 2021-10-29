// FOR MODULO USE INV FACTORIAL
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
