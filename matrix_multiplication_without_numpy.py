def assert_matrix(b):
  #check if all lists of list have same number of elements
  n = 0
  while n < len(b) - 1:
    if type(b[n]) is list and type(b[n + 1]) is list:
      if len(b[n]) == len(b[n + 1]):
          n += 1
          continue
      else:
        raise ValueError('lists of lists doesn\'t have same number of items')
    elif type(b[n] is int) and type(b[n + 1]) is int:
      n += 1
      continue
    else:
      raise TypeError('items in list must be integers and must have same type')
def matrix_mul(a, b):
  assert_matrix(a)
  assert_matrix(b)
  shape_a = (len(a), len(a[0]))
  shape_b = (len(b), len(b[0]))
  out_shape = (shape_a[0], shape_b[1])
  if type(b[0]) is list and type(a[0]) is list:
    if shape_a[1] == shape_b[0]:
      f_list = []
      for n1, i in enumerate(a):
        f = []
        for n2 in range(len(b[0])):
          s = 0
          for n3, j in enumerate(i):
            s += j * b[n3][n2]
          f.append(s)
        f_list.append(f)
      shape_f = (len(f_list), len(f_list[0]))
      return f_list
    else:
      raise ValueError(f'Can\'t multiply diffrent dimensions matrix no. of columns of {a} must be equal to no. of rows of {b}')
    #b = transpose(b)
  if type(b[0]) is int and type(a[0]) is int:
    #assert the length of both matrces
    if len(a) == 1 and len(b) == 1:
      return [a[0] * b[0]]
    else:
      raise ValueError(f'Can\'t multiply diffrent dimensions matrix no. of columns of {a} must be equal to no. of rows of {b}')
  else:
    raise ValueError(f'matrices {a} and {b} deosn\'t have same type of elements')
