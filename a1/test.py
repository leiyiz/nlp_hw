import numpy as np
import matplotlib.pyplot as plot


# question3.1
def compute3_1():
    a = np.matrix('0 2 4; 2 4 2; 3 3 1')
    a_inverse = np.linalg.inv(a)
    print('A^-1 is:')
    print(a_inverse)
    b = np.matrix('-2; -2; -4')
    print('(A^-1)b is')
    print(a_inverse * b)
    c = np.matrix('1; 1; 1')
    print('Ac is')
    print(a * c)


# question 3.2
def compute3_2_a(n, self):
    z = np.random.randn(n)
    plot.step(sorted(z), np.arange(1, n + 1) / float(n))
    plot.xlim(-3, 3)
    plot.ylabel('probability')
    plot.xlabel('observation')

    if self:
        plot.show()


def compute3_2_b(n):
    ks = [1, 8, 64, 512]
    for k in ks:
        yk = np.sum(np.sign(np.random.randn(n, k)) * np.sqrt(1. / k), axis=1)
        plot.step(sorted(yk), np.arange(1, n + 1) / float(n))

    compute3_2_a(n, False)
    plot.xlim(-3, 3)
    plot.xlabel('Observation')
    plot.ylabel('Probability')
    plot.show()


if __name__ == '__main__':
    n = 40000
    compute3_2_a(n, True)
    compute3_2_b(n)
