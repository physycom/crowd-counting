def FinalCount(A):
	row, column = A.shape()
	C = A.copy()
	C[np.arange(1, row, 2), :] = 0
	C[: np.arange(1, row, 2)] = 0
	if row % 2 == 0:
		C[row-1, :] = C[row-1, :] / 2
	if column % 2 == 0:
		C[:, column-1] = C[:, column-1] / 2
	return C.sum()