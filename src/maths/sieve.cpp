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