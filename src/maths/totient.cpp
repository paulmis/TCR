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
    (3) If aand b are relatively prime, then phi(ab) = phi(a)* phi(b)
    (4) If p is primeand k >= 1 then there are exactly p^ k / p numbers between 1 and p^ k that are divisible by p
    Therefore phi(p^ k) = p ^ k - p ^ (k - 1)
    (2) If a and m are relatively prime then a ^ (phi(m)) == = 1 (mod m)
    If m is prime then a ^ (m - 1) == = 1 (mod m)
    Therefore a ^ n == = a ^ (n mod phi(m)) (mod m)->easy a ^ n computation for big n