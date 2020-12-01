# from memory_profiler import profile
# # @profile
# def my_func():
#     a = [1] * (10 ** 6)
#     b = [2] * (2 * 10 ** 7)
#     del b
#     return a

# if __name__ == '__main__':
#     my_func()

"""
python -m memory_profiler test.py
"""

import numpy
import time

@profile
def multiply(n):
  A = numpy.random.rand(n, n)
  #time.sleep(numpy.random.randint(0, 2))
  return numpy.matrix(A) ** 2

for n in 2 ** numpy.arange(0, 10):
  multiply(n)

"""
kernprof.py -l -v test.py
"""