vi ans;
vvi combinations;

void gen(int n, int k, int idx, bool rev) {
	if (k > n || k < 0) return;
	if (!n) {
		for (int i = 0; i < idx; ++i) {
			combinations.push_back(vi());
			if (ans[i])
				combinations[combinations.size()].pb(i + 1);
		}
		return;
	}
	ans[idx] = rev;
	gen(n - 1, k - rev, idx + 1, false);
	ans[idx] = !rev;
	gen(n - 1, k - !rev, idx + 1, true);
}

// Steinhaus–Johnson–Trotter algorithm [generate all combinations by swapping one element each iteration] O(n * (n choose k))
void adjecentSwapCombinations(int n, int k) {
	ans.resize(n);
	gen(n, k, 0, false);
}