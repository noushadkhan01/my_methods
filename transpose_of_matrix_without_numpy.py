def transpose_without_numpy(matrix):
  #make a copy of original matrix
  copy = matrix
  #create for loop and apply basic method of transpose 
  #copy_i_{j} = matrix_i_{j}
  for n in range(len(matrix)):
    for n2 in range(len(matrix)):
      copy[n][n2] = copy[n2][n]
  return copy
