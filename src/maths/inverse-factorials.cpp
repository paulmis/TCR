fac[0] = ifac[0] = ifac[1] = 1;
for (int i = 1; i < maxn; i++)
    fac[i] = fac[i - 1] * i % mod;
ifac[maxn - 1] = mpow(fac[maxn - 1], mod - 2);
for (int i = maxn - 2; i > 1; i--)
    ifac[i] = ifac[i + 1] * (i + 1) % mod;