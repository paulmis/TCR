// Levenshtein distance [min. edits to transform A to B] O(n^2)
int levenshtein(string& A, string& B)
{
	int a = A.size(), b = B.size();
	vvi d(a, vi(b, 0));
	for (int i = 1; i <= a; ++i) d[i][0] = i;
	for (int i = 1; i <= b; ++i) d[0][i] = i;
	for (int i = 1; i <= a; ++i)
		for (int j = 1; j <= b; ++j)
			d[i][j] = std::min({ d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (A[i - 1] == B[j - 1] ? 0 : 1) });
	return d[a][b];
}