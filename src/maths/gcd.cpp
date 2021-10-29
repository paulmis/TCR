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