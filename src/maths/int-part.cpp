// # of distinct ways to sum integers to n
ll intPart(int n) {
	if (n < 2) return P[n] = 1;
	if (!P[n])
		for (int k = 1; (k * (3 * k - 1)) / 2 <= n; k++)
			for (int i = 1; i < 4 && k * (3 * k + i - 2) / 2 <= n; i += 2)
				P[n] += ((k & 1 ? 1 : -1) * intPart(n - (k * (3 * k + i - 2) / 2)));
	return P[n];
}