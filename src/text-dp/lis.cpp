// Longest increasing subsequence O(nlogn)
int lis(vi& arr) {
	int n = arr.size();
	vi d(n + 1, INT_MAX);
	d[0] = -INT_MAX;
	for (int i = 0; i < n; i++) {
		int j = upper_bound(d.begin(), d.end(), arr[i]) - d.begin();
		if (d[j - 1] < arr[i] && arr[i] < d[j]) d[j] = arr[i];
	}

	int ans = 0;
	for (int i = 0; i <= n; i++)
		if (d[i] < INT_MAX)
			ans = i;
	return ans;
}

It is also possible to restore the subsequence using this approach.This time we have to maintain two auxiliary arrays.One that tells us the index of the elements in d[].And again we have to create an array of "ancestors" p[i].p[i] will be the index of the previous element for the optimal subsequence ending in element i.
It is easy to maintain these two arrays in the course of iteration over the array a[] alongside the computations of d[]. And at the end it is not difficult to restore the desired subsequence using these arrays.