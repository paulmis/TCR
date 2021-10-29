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