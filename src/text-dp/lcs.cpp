// Longest common subsequence O(n^2)
string LCseq(string& A, string& B) {
    string str;
    int a = A.size(), b = B.size();
    vvi DP(a + 1, vi(b + 1, 0));
    for (int i = 1; i <= a; i++)
        for (int j = 1; j <= b; j++)
        {
            if (A[i - 1] == B[j - 1]) DP[i][j] = DP[i - 1][j - 1] + 1;
            else                      DP[i][j] = max(DP[i - 1][j], DP[i][j - 1]);
        }
    for (int i = a - 1, j = b - 1; i >= 0 && j >= 0; ) {
        if (A[i] == B[j]) {
            str += A[i];
            i--, j--;
        } else {
            if (DP[i + 1][j] < DP[i][j + 1]) i--;
            else                             j--;
        }
    }
    reverse(all(str));
    return str;
}