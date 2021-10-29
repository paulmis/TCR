// O(n)
vi computepi(string& s) {
	int n = s.size();
	vi p(n);
	for (int i = 1; i < n; i++) {
		int j = p[i - 1];
		while (j > 0 && s[i] != s[j]) j = p[j - 1];
		if (s[i] == s[j]) j++;
		p[i] = j;
	}
	return p;
}

// Find s in t O(t)
int kmp(string& s, string& t) {
	vi p = computepi(s);
	for (int i = 0, j = 0; i < t.size(); i++) {
		while (j > 0 && t[i] != s[j]) j = p[j - 1];
		if (t[i] == s[j] && ++j == s.size()) return i - j + 1;
	}
	return -1;
}

The prefix function for this string is defined as an array p of length n, where p[i] is the length of the longest proper prefix of the substring s[0..i] which is also a suffix of this substring.A proper prefix of a string is a prefix that is not equal to the string itself.By definition, p[0] = 0.
Counting the number of occurrences of each prefix
	vi ans(n + 1);
	for (int i = 0; i < n; i++)		ans[p[i]]++;
	for (int i = n - 1; i > 0; i--) ans[p[i - 1]] += ans[i];
	for (int i = 0; i <= n; i++)	ans[i]++;