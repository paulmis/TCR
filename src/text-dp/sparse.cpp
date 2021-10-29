int ts, k;
vvll st;
vi lt;

// ONLY STATIC
// O(nlogn) build O(1) RMQ
void build(vll& arr) {
	k = 1; ts = arr.size();
	lt.assign(arr.size(), 0);
	for (int i = 2; i <= ts; i++)
		lt[i] = lt[i / 2] + 1;
	for (int p = 1; p < arr.size(); p <<= 1, k++);
	st.resize(ts + 1, vll(k + 1, 0));
	for (int i = 0; i < ts; i++)
		st[i][0] = arr[i];
	for (int j = 1; j <= k; j++)
		for (int i = 0; i + (1 << j) <= ts; i++)
			st[i][j] = min(st[i][j - 1], st[i + (1 << (j - 1))][j - 1]);
}

ll min(int l, int r) { // inclusive
	int j = lt[r - l + 1];
	int minimum = min(st[l][j], st[r - (1 << j) + 1][j]);
}
