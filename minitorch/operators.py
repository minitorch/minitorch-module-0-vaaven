"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


def mul(x, y):
    return x * y

def id(x):
    return x

def add(x, y):
    return x + y

def neg(x):
    return -x

def lt(x, y):
    return x < y

def eq(x, y):
    return x == y

def max(x, y):
    return x if x > y else y

def is_close(x, y):
    return abs(x - y) < 1e-2

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def relu(x):
    return x if x > 0 else 0

def log(x):
    return math.log(x)

def exp(x):
    return math.exp(x)

def log_back(x, y):
    return 1.0 / x

def inv(x):
    return 1.0 / x

def inv_back(x, y):
    return -1.0 / (x ** 2) * y

def relu_back(x, y):
    return y if x > 0 else 0



# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists

def map(f, ls):
    return [f(x) for x in ls]

def zipWith(f, ls1, ls2):
    return [f(x, y) for x, y in zip(ls1, ls2)]

def reduce(f, ls):
    ans = ls[0]
    for x in ls[1:]:
        ans = f(ans, x)
    return ans

def negList(ls):
    return map(neg, ls)

def addLists(ls1, ls2):
    return zipWith(add, ls1, ls2)

def sum(ls):
    return reduce(add, ls) if ls else 0

def prod(ls):
    return reduce(mul, ls) if ls else 1