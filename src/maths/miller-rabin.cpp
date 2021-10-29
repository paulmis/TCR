// Miller-rabin primarility test (a values for ll)
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