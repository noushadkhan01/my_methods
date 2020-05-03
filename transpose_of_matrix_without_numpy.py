def transpose_without_numpy(matrix):
  if type(copy) is np.ndarray:
    copy = matrix.copy()
  else:
    copy = matrix
  for n in range(len(matrix)):
    for n2 in range(len(matrix)):
      copy[n][n2] = matrix[n2][n]
  return copy
