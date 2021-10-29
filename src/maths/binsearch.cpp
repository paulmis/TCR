int l = 0, r = arr.size() - 1;
while (l <= r) {
    int m = l + (r - l) / 2;
    if (table[m] == key) return true;
    if (table[m] > key)  r = m - 1;
    else                 l = m + 1;
}
return false;
