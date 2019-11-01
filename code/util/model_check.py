def count_num_params(model):
  sum = 0
  for each in list(model.parameters()):
    size = list(each.size())
    tmp = 1
    for i in size:
      tmp = tmp * i
    sum += tmp
  return sum
