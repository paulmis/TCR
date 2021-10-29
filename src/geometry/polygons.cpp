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