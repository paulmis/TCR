Let A be the adjacency matrix of the graph : Au, v is the number of edges between uand v.Let D be the degree matrix of the graph : a diagonal matrix with Du, u being the degree of vertex u(including multiple edgesand loops - edges which connect vertex u with itself).

The Laplacian matrix of the graph is defined as L = D - A.According to Kirchhoff's theorem, all cofactors of this matrix are equal to each other, and they are equal to the number of spanning trees of the graph. The (i,j) cofactor of a matrix is the product of (-1)i+j with the determinant of the matrix that you get after removing the i-th row and j-th column. So you can, for example, delete the last row and last column of the matrix L, and the absolute value of the determinant of the resulting matrix will give you the number of spanning trees.