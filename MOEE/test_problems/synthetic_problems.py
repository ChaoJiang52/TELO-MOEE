"""A selection of synthetic test problems.

Each test problem can be instantiated and called in the following way:
>> f_class = WangFretias
>> f = f_class()
>> f(numpy.array([0.9]))
array([-4.])

The test problems can either be called with a single value to evaluate
(as above), or a numpy.ndarray of N values with shape (N, D) where D is the
problem dimensionality:
>> f_class = WangFretias
>> f = f_class()
>> X = numpy.array([[0.  ], [0.25], [0.5 ], [0.75],[1.  ]])
>> f(X)
array([-1.21306132e+00, -6.49304935e-01, -6.70925256e-04, -1.33831722e-09,
       -5.15428572e-18])

The test problems each have the following attributes:
    dim : int
        Dimensionality (D) of the problem
    lb : (D, ) numpy.ndarray
        Lower bound for each of the problem's D dimensions
    ub : (D, ) numpy.ndarray
        Upper bound for each of the problem's D dimensions
    xopt : (D, ) numpy.ndarray or (N, D) numpy.ndarray
        Location of the (1 or N) optima
    yopt : float or (1, 1) numpy.ndarray
        Function value of the optimum (i.e. f(f.xopt))
    cf : callable or None
        A constraint function that takes in a decision vector and returns a
        boolean value indicating whether it passes the constraint function
        (True) or violates it (False). If the test problem has no constraint
        function, then cf should be set to None.
"""
import numpy as np
import cocoex

suite = cocoex.Suite('bbob', '', '')


class Attractive_Sector:
    """
       f6 Schwefel FUNCTION http://numbbo.github.io/coco/testsuites/bbob
    """

    def __init__(self):
        self.dim = 10
        self.lb = np.full(self.dim, -5)
        self.ub = np.full(self.dim, 5)

        self.problem = suite.get_problem_by_function_dimension_instance(6, 10, 1)
        self.problem._best_parameter('print')
        self.xopt = np.loadtxt('._bbob_problem_best_parameter.txt')
        self.yopt = 35.9
        # self.yopt = self.problem(self.xopt)  # 35.9
        self.cf = None

    def __call__(self, x):
        x = np.atleast_2d(x)
        return np.array([self.problem(i) for i in x]).ravel()


class Ellipsoidal:
    """
       f10 Ellipsoidal FUNCTION http://numbbo.github.io/coco/testsuites/bbob
    """

    def __init__(self):
        self.dim = 10
        self.lb = np.full(self.dim, -5)
        self.ub = np.full(self.dim, 5)

        self.problem = suite.get_problem_by_function_dimension_instance(10, 10, 1)
        self.problem._best_parameter('print')
        self.xopt = np.loadtxt('._bbob_problem_best_parameter.txt')
        self.yopt = -54.94
        # self.yopt = self.problem(self.xopt)  # -54.94
        self.cf = None

    def __call__(self, x):
        x = np.atleast_2d(x)
        return np.array([self.problem(i) for i in x]).ravel()


class Schwefel:
    """
        f20 Schwefel FUNCTION http://numbbo.github.io/coco/testsuites/bbob
    """

    def __init__(self):
        self.dim = 10
        self.lb = np.full(self.dim, -5)
        self.ub = np.full(self.dim, 5)

        self.problem = suite.get_problem_by_function_dimension_instance(20, 10, 1)
        self.problem._best_parameter('print')
        self.xopt = np.loadtxt('._bbob_problem_best_parameter.txt')
        self.yopt = -546.5
        # self.yopt = self.problem(self.xopt)  # -546.5
        self.cf = None

    def __call__(self, x):
        x = np.atleast_2d(x)
        return np.array([self.problem(i) for i in x]).ravel()


class GRIEWANK_20:
    """
    GRIEWANK FUNCTION https://www.sfu.ca/~ssurjano/griewank.html

    """

    def __init__(self):
        self.dim = 20
        self.lb = np.full(self.dim, -600)
        self.ub = np.full(self.dim, 600)

        self.xopt = np.full(self.dim, 0)
        self.yopt = self.__call__(self.xopt)

        self.cf = None

    def __call__(self, x):
        x = np.atleast_2d(x)
        sum1 = np.sum(x[np.newaxis, :] ** 2 / 4000, axis=2)
        prod = np.prod(np.cos(x / np.sqrt(np.arange(1, self.dim + 1))), axis=1)

        val = sum1 - prod + 1
        return val.ravel()


class GRIEWANK_10:
    """
    GRIEWANK FUNCTION https://www.sfu.ca/~ssurjano/griewank.html

    """

    def __init__(self):
        self.dim = 10
        self.lb = np.full(self.dim, -600)
        self.ub = np.full(self.dim, 600)

        self.xopt = np.full(self.dim, 0)
        self.yopt = self.__call__(self.xopt)

        self.cf = None

    def __call__(self, x):
        x = np.atleast_2d(x)
        sum1 = np.sum(x[np.newaxis, :] ** 2 / 4000, axis=2)
        prod = np.prod(np.cos(x / np.sqrt(np.arange(1, self.dim + 1))), axis=1)

        val = sum1 - prod + 1
        return val.ravel()


class GRIEWANK_2:
    """
    GRIEWANK FUNCTION https://www.sfu.ca/~ssurjano/griewank.html

    """

    def __init__(self):
        self.dim = 2
        self.lb = np.full(self.dim, -600)
        self.ub = np.full(self.dim, 600)

        self.xopt = np.full(self.dim, 0)
        self.yopt = self.__call__(self.xopt)

        self.cf = None

    def __call__(self, x):
        x = np.atleast_2d(x)
        sum1 = np.sum(x[np.newaxis, :] ** 2 / 4000, axis=2)
        prod = np.prod(np.cos(x / np.sqrt(np.arange(1, self.dim + 1))), axis=1)

        val = sum1 - prod + 1
        return val.ravel()


class ACKLEY_20:
    """
    ACKLEY FUNCTION https://www.sfu.ca/~ssurjano/ackley.html

    """

    def __init__(self):
        self.dim = 20
        self.lb = np.full(self.dim, -32.768)
        self.ub = np.full(self.dim, 32.768)

        self.xopt = np.full(self.dim, 0)
        self.yopt = self.__call__(self.xopt)

        self.cf = None

    def __call__(self, x):
        x = np.atleast_2d(x)
        a = 20
        b = 0.2
        c = 2 * np.pi
        sum1 = np.sum(x[np.newaxis, :] ** 2, axis=2)
        sum2 = np.sum(np.cos(c * x[np.newaxis, :]), axis=2)
        term1 = -a * np.exp(-b * np.sqrt(sum1 / self.dim))
        term2 = -np.exp(sum2 / self.dim)
        val = term1 + term2 + a + np.exp(1)
        return val.ravel()


class ACKLEY_10:
    """
    ACKLEY FUNCTION https://www.sfu.ca/~ssurjano/ackley.html

    """

    def __init__(self):
        self.dim = 10
        self.lb = np.full(self.dim, -32.768)
        self.ub = np.full(self.dim, 32.768)

        self.xopt = np.full(self.dim, 0)
        self.yopt = self.__call__(self.xopt)

        self.cf = None

    def __call__(self, x):
        x = np.atleast_2d(x)
        a = 20
        b = 0.2
        c = 2 * np.pi
        sum1 = np.sum(x[np.newaxis, :] ** 2, axis=2)
        sum2 = np.sum(np.cos(c * x[np.newaxis, :]), axis=2)
        term1 = -a * np.exp(-b * np.sqrt(sum1 / self.dim))
        term2 = -np.exp(sum2 / self.dim)
        val = term1 + term2 + a + np.exp(1)
        return val.ravel()


class ACKLEY_2:
    """
    ACKLEY FUNCTION https://www.sfu.ca/~ssurjano/ackley.html

    """

    def __init__(self):
        self.dim = 2
        self.lb = np.full(self.dim, -32.768)
        self.ub = np.full(self.dim, 32.768)

        self.xopt = np.full(self.dim, 0)
        self.yopt = self.__call__(self.xopt)

        self.cf = None

    def __call__(self, x):
        x = np.atleast_2d(x)
        a = 20
        b = 0.2
        c = 2 * np.pi
        sum1 = np.sum(x[np.newaxis, :] ** 2, axis=2)
        sum2 = np.sum(np.cos(c * x[np.newaxis, :]), axis=2)
        term1 = -a * np.exp(-b * np.sqrt(sum1 / self.dim))
        term2 = -np.exp(sum2 / self.dim)
        val = term1 + term2 + a + np.exp(1)
        return val.ravel()


class LEVY_10:
    """
    LEVY FUNCTION https://www.sfu.ca/~ssurjano/levy.html

    """

    def __init__(self):
        self.dim = 2
        self.lb = np.full(self.dim, -10)
        self.ub = np.full(self.dim, 10)

        self.xopt = np.full(self.dim, 1)
        self.yopt = self.__call__(self.xopt)

        self.cf = None

    def __call__(self, x):
        x = np.atleast_2d(x)
        w = 1 + (x - 1) / 4

        term1 = (np.sin(np.pi * w[:, 0])) ** 2
        term3 = (w[:, -1] - 1) ** 2 * (1 + (np.sin(2 * np.pi * w[:, -1])) ** 2)
        term2 = 0
        for w_i in w[:, :-1]:
            term2 += (w_i - 1) ** 2 * (1 + 10 * (np.sin(np.pi * w_i + 1)) ** 2)

        val = term1 + term2 + term3
        return val.ravel()


class LEVY_2:
    """
    LEVY FUNCTION https://www.sfu.ca/~ssurjano/levy.html

    """

    def __init__(self):
        self.dim = 2
        self.lb = np.full(self.dim, -10)
        self.ub = np.full(self.dim, 10)

        self.xopt = np.full(self.dim, 1)
        self.yopt = self.__call__(self.xopt)

        self.cf = None

    def __call__(self, x):
        x = np.atleast_2d(x)
        w = 1 + (x - 1) / 4

        term1 = (np.sin(np.pi * w[:, 0])) ** 2
        term3 = (w[:, -1] - 1) ** 2 * (1 + (np.sin(2 * np.pi * w[:, -1])) ** 2)
        term2 = 0
        for w_i in w[:, :-1]:
            term2 += (w_i - 1) ** 2 * (1 + 10 * (np.sin(np.pi * w_i + 1)) ** 2)

        val = term1 + term2 + term3
        return val.ravel()


class LEVY13:
    """
    LEVY FUNCTION N. 13 https://www.sfu.ca/~ssurjano/levy13.html

    """

    def __init__(self):
        self.dim = 2
        self.lb = np.full(self.dim, -10)
        self.ub = np.full(self.dim, 10)

        self.xopt = np.full(self.dim, 1)
        self.yopt = self.__call__(self.xopt)

        self.cf = None

    def __call__(self, x):
        x = np.atleast_2d(x)
        term1 = (np.sin(3 * np.pi * x[:, 0])) ** 2
        term2 = (x[:, 0] - 1) ** 2 * (1 + (np.sin(3 * np.pi * x[:, 1])) ** 2)
        term3 = (x[:, 1] - 1) ** 2 * (1 + (np.sin(2 * np.pi * x[:, 1])) ** 2)

        val = term1 + term2 + term3
        return val.ravel()


class WangFreitas:
    """WangFreitas [1]_ test function with deceptive optimum.

    .. math::
        f(x) = - (2 e^{-0.5 (x - a)^2) / theta_1}
                  + 4 e^{-0.5 (x - b)^2) / theta_2}
                  + epsilon )

    where:

    .. math::
        a = 0.1
        b = 0.9
        theta_1 = 0.1
        theta_2 = 0.01
        epsilon = 0

    References
    ----------
    .. [1] Ziyu Wang and Nando de Freitas. 2014.
       Theoretical analysis of Bayesian optimisation with unknown Gaussian
       process hyper-parameters. arXiv:1406.7758
    """

    def __init__(self):
        self.dim = 1
        self.lb = np.zeros(self.dim)
        self.ub = np.ones(self.dim)

        # function parameters
        self.a = 0.1
        self.b = 0.9
        self.theta_1 = 0.1
        self.theta_2 = 0.01
        self.epsilon = 0

        self.xopt = np.array([0.9])
        self.yopt = self.__call__(self.xopt)

        self.cf = None

    def __call__(self, x):
        x = np.atleast_2d(x)
        val = -(
                2 * np.exp(-0.5 * (self.a - x) ** 2 / self.theta_1 ** 2)
                + 4 * np.exp(-0.5 * (self.b - x) ** 2 / self.theta_2 ** 2)
                + self.epsilon
        )
        return val.ravel()


class Branin:
    """Branin, or Branin-Hoo, test function.
    See: https://www.sfu.ca/~ssurjano/branin.html

    .. math::
        f(x) = a(x_2 - b x_1^2 + c x_1 - r)^2 + s(1 - t) cos(x_1) + s

    where:

    .. math::
        a = 1
        b = 5.1 / (4 pi^2)
        c = 5 / pi
        r = 6
        s = 10
        t = 1 / (8 pi)
    """

    def __init__(self):
        self.dim = 2
        self.lb = np.array([-5, 0])
        self.ub = np.array([10, 15])

        # function parameters
        self.a = 1
        self.b = 5.1 / (4 * np.pi ** 2)
        self.c = 5 / np.pi
        self.r = 6
        self.s = 10
        self.t = 1 / (8 * np.pi)

        self.xopt = np.array([[-np.pi, 12.275], [np.pi, 2.275], [9.42478, 2.475]])
        self.yopt = self.__call__(self.xopt[0, :])  # 0.397887

        self.cf = None

    def __call__(self, x):
        x = np.atleast_2d(x)
        val = (
                self.a * (x[:, 1] - self.b * x[:, 0] ** 2 + self.c * x[:, 0] - self.r) ** 2
                + self.s * (1 - self.t) * np.cos(x[:, 0])
                + self.s
        )
        return val.ravel()


class BraninForrester:
    """Branin test function with modifications from Forrester et al. [1]_.
    The change from Forrester et al. modifies the function to only have one
    global minimum, making it more representative of engineering functions.
    See: https://www.sfu.ca/~ssurjano/branin.html

    .. math::
        f(x) = a(x_2 - b x_1^2 + c x_1 - r)^2 + s(1 - t) cos(x_1) + s + 5 x_1

    where:

    .. math::
        a = 1
        b = 5.1 / (4 pi^2)
        c = 5 / pi
        r = 6
        s = 10
        t = 1 / (8 pi)

    References
    ----------
    .. [1] Forrester, A., Sobester, A., & Keane, A. (2008).
       Engineering design via surrogate modelling: a practical guide. Wiley.
    """

    def __init__(self):
        self.dim = 2
        self.lb = np.array([-5, 0])
        self.ub = np.array([10, 15])

        # function parameters
        self.a = 1.0
        self.b = 5.1 / (4 * np.pi ** 2)
        self.c = 5.0 / np.pi
        self.r = 6.0
        self.s = 10.0
        self.t = 1.0 / (8 * np.pi)

        self.xopt = np.array([-3.689, 13.629])
        self.yopt = self.__call__(self.xopt)  # -16.64402  -16.64402117

        self.cf = None

    def __call__(self, x):
        x = np.atleast_2d(x)
        val = (
                self.a * (x[:, 1] - self.b * x[:, 0] ** 2 + self.c * x[:, 0] - self.r) ** 2
                + self.s * (1 - self.t) * np.cos(x[:, 0])
                + self.s
                + 5 * x[:, 0]
        )
        return val.ravel()


class logGoldsteinPrice:
    """log(GoldSteinPrice) test function
    See: http://www.sfu.ca/~ssurjano/goldpr.html

    .. math::
        f(x) = log((1 + (x_1 + x_2 +1 )^2 (19 - 14 x_1 + 3 x_1^2 - 14 x_2 + 6 x_1 x_2 + 3 x_2^2))
                   (30 + (2 x_1 - 3 x_2)^2 (18 - 32 x_1 + 12 x_1^2 + 48 x_2 - 36 x_1 x_2 + 27 x_2^2))
                  )
    """

    def __init__(self):
        self.dim = 2
        self.lb = np.full(self.dim, -2)
        self.ub = np.full(self.dim, 2)

        self.xopt = np.array([0, -1])
        self.yopt = self.__call__(self.xopt)

        self.cf = None

    def __call__(self, x):
        x = np.atleast_2d(x)
        val = np.log(
            (
                    1
                    + (x[:, 0] + x[:, 1] + 1) ** 2
                    * (
                            19
                            - 14 * x[:, 0]
                            + 3 * x[:, 0] ** 2
                            - 14 * x[:, 1]
                            + 6 * x[:, 0] * x[:, 1]
                            + 3 * x[:, 1] ** 2
                    )
            )
            * (
                    30
                    + (2 * x[:, 0] - 3 * x[:, 1]) ** 2
                    * (
                            18
                            - 32 * x[:, 0]
                            + 12 * x[:, 0] ** 2
                            + 48 * x[:, 1]
                            - 36 * x[:, 0] * x[:, 1]
                            + 27 * x[:, 1] ** 2
                    )
            )
        )
        return val.ravel()


class GoldsteinPrice:
    """
    Goldstein-Price Function
    See: http://www.sfu.ca/~ssurjano/goldpr.html

    .. math::
        f(x) = (1 + (x_1 + x_2 +1 )^2 (19 - 14 x_1 + 3 x_1^2 - 14 x_2 + 6 x_1 x_2 + 3 x_2^2))
                (30 + (2 x_1 - 3 x_2)^2 (18 - 32 x_1 + 12 x_1^2 + 48 x_2 - 36 x_1 x_2 + 27 x_2^2))
    """

    def __init__(self):
        self.dim = 2
        self.lb = np.full(self.dim, -2.0)
        self.ub = np.full(self.dim, 2.0)

        # optimum value(s)
        self.xopt = np.array([0.0, -1.0])
        self.yopt = self.__call__(self.xopt)  # 3

        # constraint
        self.cf = None

    def __call__(self, x):
        x = np.atleast_2d(x)

        term1 = 1.0 + (x[:, 0] + x[:, 1] + 1) ** 2 * (
                19.0
                - 14.0 * x[:, 0]
                + 3.0 * x[:, 0] ** 2
                - 14.0 * x[:, 1]
                + 6.0 * x[:, 0] * x[:, 1]
                + 3.0 * x[:, 1] ** 2
        )
        term2 = 30 + (2.0 * x[:, 0] - 3.0 * x[:, 1]) ** 2 * (
                18.0
                - 32.0 * x[:, 0]
                + 12.0 * x[:, 0] ** 2
                + 48.0 * x[:, 1]
                - 36.0 * x[:, 0] * x[:, 1]
                + 27.0 * x[:, 1] ** 2
        )
        val = term1 * term2

        return val.ravel()


class Cosines:
    """Cosines test function by González et al. [1]_.
    See: http://proceedings.mlr.press/v51/gonzalez16b-supp.pdf

    .. math::
        f(x)= - (1 - sum_{i=1}^2 g(x_i) - r(x_i))

    where:

    .. math::
        g(x_i) = (1.6 x_i - 0.5)^2
        r(x_i) = 0.3 cos (3 pi (1.6 x_i - 0.5) )

    References
    ----------
    .. [1] Javier González, Michael Osborne, and Neil Lawrence. 2016.
       GLASSES: Relieving the myopia of Bayesian optimisation.
       In Proceedings of the 19th International Conference on Artificial
       Intelligence and Statistics, Vol. 51. PMLR, 790–799.
    """

    def __init__(self):
        self.dim = 2
        self.lb = np.full(self.dim, 0)
        self.ub = np.full(self.dim, 5)

        self.xopt = np.array([0.3125, 0.3125])
        self.yopt = -1.6

        self.cf = None

    def __call__(self, x):
        x = np.atleast_2d(x)
        g = (1.6 * x - 0.5) ** 2
        r = 0.3 * np.cos(3 * np.pi * (1.6 * x - 0.5))
        val = -(1 - np.sum(g - r, axis=1))
        return val.ravel()


class logSixHumpCamel:
    """log(Six-Hump Camel) test function.
    See: https://www.sfu.ca/~ssurjano/camel6.html

    .. math::
        f(x)= log( (4 - 2.1 x_1^2 + (x_1^4 / 3)) x_1^2 + x_1 x_2
                  + (-4 + 4 x_2^2) x_2^2
                  + 1.0316 + 1e-4)
    """

    def __init__(self):
        self.dim = 2
        self.lb = np.array([-3, -2])
        self.ub = np.array([3, 2])

        self.xopt = np.array([[0.0898, -0.7126], [-0.0898, 0.7126]])
        self.yopt = self.__call__(self.xopt[0])

        self.cf = None

    def __call__(self, x):
        x = np.atleast_2d(x)
        val = np.log(
            (4 - 2.1 * x[:, 0] ** 2 + (x[:, 0] ** 4 / 3)) * x[:, 0] ** 2
            + x[:, 0] * x[:, 1]
            + (-4 + 4 * x[:, 1] ** 2) * x[:, 1] ** 2
            + 1.0316
            + 1e-4
        )
        return val.ravel()


class SixHumpCamel:
    """
    Six-Hump Camel Function
    See: http://www.sfu.ca/~ssurjano/camel6.html

    .. math::
        f(x)= (4 - 2.1 x_1^2 + (x_1^4 / 3)) x_1^2 + x_1 x_2
              + (-4 + 4 x_2^2) x_2^2
    """

    def __init__(self):
        self.dim = 2
        self.lb = np.array([-3.0, -2.0])
        self.ub = np.array([3.0, 2.0])

        # optimum value(s)
        self.xopt = np.array([[0.0898, -0.7126], [-0.0898, 0.7126]])
        self.yopt = self.__call__(self.xopt[0, :])  # -1.03162845

        # constraint
        self.cf = None

    def __call__(self, x):
        x = np.atleast_2d(x)

        xsqr = np.square(x)

        term1 = (4.0 - 2.1 * xsqr[:, 0] + x[:, 0] ** 4 / 3.0) * xsqr[:, 0]
        term2 = np.prod(x, axis=1) + (-4.0 + 4.0 * xsqr[:, 1]) * xsqr[:, 1]

        val = term1 + term2

        return val.ravel()


class logHartmann6:
    """log(Hartmann6) 6-dimensional test problem.
    See: https://www.sfu.ca/~ssurjano/hart6.html for the formula.
    """

    def __init__(self):
        self.dim = 6
        self.lb = np.zeros(6)
        self.ub = np.ones(6)

        self.A = np.array(
            [
                [10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
                [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
                [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
                [17.0, 8.0, 0.05, 10.0, 0.1, 14.0],
            ]
        )
        self.P = np.array(
            [
                [0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
                [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
                [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.665],
                [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381],
            ]
        )
        self.alpha = np.array([[1.0], [1.2], [3.0], [3.2]])

        self.xopt = np.array(
            [[0.201690, 0.150011, 0.476874, 0.275332, 0.311652, 0.657300]]
        )
        self.yopt = -1.20067779

        self.cf = None

    def __call__(self, x):
        x = np.atleast_2d(x)
        val = np.zeros((x.shape[0], 1))

        for i in range(x.shape[0]):
            x_val = x[i, :]
            B = (x_val[np.newaxis, :] - self.P) ** 2
            val[i, :] = -np.log(
                np.dot(self.alpha.T, np.exp(-np.diag(np.dot(self.A, B.T))))
            )

        return val.ravel()


class Hartmann6:
    """
    Hartmann 6D Function
    See: https://www.sfu.ca/~ssurjano/hart6.html for the formula.
    """

    def __init__(self):
        self.dim = 6
        self.lb = np.zeros(self.dim)
        self.ub = np.ones(self.dim)

        # function parameters
        self.alpha = np.array([1.0, 1.2, 3.0, 3.2])
        self.A = np.array(
            [
                [10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
                [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
                [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
                [17.0, 8.0, 0.05, 10.0, 0.1, 14.0],
            ]
        )

        self.P = np.array(
            [
                [0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
                [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
                [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
                [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381],
            ]
        )

        # optimum value(s)
        self.xopt = np.array(
            [[0.201690, 0.150011, 0.476874, 0.275332, 0.311652, 0.657300]]
        )
        self.yopt = self.__call__(self.xopt)  # -3.32236801

        # constraint
        self.cf = None

    def __call__(self, x):
        x = np.atleast_2d(x)

        term1 = np.sum(
            self.A[np.newaxis, :, :]
            * (x[:, np.newaxis, :] - self.P[np.newaxis, :, :]) ** 2,
            axis=2,
        )
        term2 = -np.sum(self.alpha[np.newaxis, :] * np.exp(-term1), axis=1)

        val = term2

        return val.ravel()


class logGSobol:
    """ 10D gSobol function by González et al. [1]_.

    .. math::
        f(x)= log( prod_{i=1}^D (|4 x_i - 2| + a_1) / (1 + a_i) )

    where:

    .. math::
        a_i = 1

    References
    ----------
    .. [1] Javier González, Zhenwen Dai, Philipp Hennig, and Neil Lawrence.
       2016. Batch Bayesian optimization via local penalization.
       In Proceedings of the 19th International Conference on Artificial
       Intelligence and Statistics, Vol. 51. PMLR, 648–657.
    """

    def __init__(self):
        self.dim = 10
        self.lb = np.full(self.dim, -5)
        self.ub = np.full(self.dim, 5)

        self.a = np.ones(self.dim)

        self.xopt = np.full(self.dim, 0.5)
        self.yopt = self.__call__(self.xopt)  # -6.93147181

        self.cf = None

    def __call__(self, x):
        x = np.atleast_2d(x)
        val = np.log(np.prod((np.abs(4 * x - 2) + self.a[0]) / (1 + self.a[0]), axis=1))
        return val.ravel()


class GSobol:
    """ 10D gSobol function by González et al. [1]_.

    .. math::
        f(x)= prod_{i=1}^D (|4 x_i - 2| + a_1) / (1 + a_i)

    where:

    .. math::
        a_i = 1

    References
    ----------
    .. [1] Javier González, Zhenwen Dai, Philipp Hennig, and Neil Lawrence.
       2016. Batch Bayesian optimization via local penalization.
       In Proceedings of the 19th International Conference on Artificial
       Intelligence and Statistics, Vol. 51. PMLR, 648–657.
    """

    def __init__(self):
        self.dim = 10
        self.lb = np.full(self.dim, -5)
        self.ub = np.full(self.dim, 5)

        self.a = np.ones(self.dim)

        self.xopt = np.full(self.dim, 0.5)
        self.yopt = self.__call__(self.xopt)

        self.cf = None

    def __call__(self, x):
        x = np.atleast_2d(x)
        val = np.prod((np.abs(4 * x - 2) + self.a[0]) / (1 + self.a[0]), axis=1)
        return val.ravel()


class logRosenbrock:
    """log(Rosenbrock) test function.
    See: https://www.sfu.ca/~ssurjano/rosen.html

    .. math::
        f(x) = log( sum_{i=1}^{d-1} (100 (x_{i+1} - x_i^2)^2 + (x_i - 1)^2)
                   + 0.5)
    """

    def __init__(self):
        self.dim = 10
        self.lb = np.full(self.dim, -5)
        self.ub = np.full(self.dim, 10)

        self.xopt = np.ones(self.dim)
        self.yopt = self.__call__(self.xopt)

        self.cf = None

    def __call__(self, x):
        x = np.atleast_2d(x)
        last = (x - 1) ** 2
        sqr = x ** 2
        val = np.log(
            np.sum(100 * (x[:, 1:] - sqr[:, :-1]) ** 2 + last[:, :-1], axis=1) + 0.5
        )
        return val.ravel()


class Rosenbrock:
    """
    Rosenbrock Function
    See: http://www.sfu.ca/~ssurjano/rosen.html

    .. math::
        f(x) = sum_{i=1}^{d-1} (100 (x_{i+1} - x_i^2)^2 + (x_i - 1)^2)
    """

    def __init__(self):
        self.dim = 10
        self.lb = np.full(self.dim, -5.0)
        self.ub = np.full(self.dim, 10.0)

        # optimum value(s)
        self.xopt = np.ones(self.dim)
        self.yopt = self.__call__(self.xopt)

        # constraint
        self.cf = None

    def __call__(self, x):
        x = np.atleast_2d(x)

        term1 = 100.0 * (x[:, 1:] - x[:, :-1] ** 2) ** 2
        term2 = (x[:, :-1] - 1) ** 2
        term3 = np.sum(term1 + term2, axis=1)

        val = term3

        return val.ravel()


class logStyblinskiTang:
    """log(StyblinskiTang) test function
    See: https://www.sfu.ca/~ssurjano/stybtang.html

    .. math::
        f(x) = log( (1/2) sum_{i=1}^D (x_i^4 - 16 x_i^2 + 5 x_i) + 400)
    """

    def __init__(self):
        self.dim = 10
        self.lb = np.full(self.dim, -5)
        self.ub = np.full(self.dim, 5)

        self.xopt = np.full(self.dim, -2.903534)
        self.yopt = self.__call__(self.xopt)

        self.cf = None

    def __call__(self, x):
        x = np.atleast_2d(x)
        val = np.log(0.5 * np.sum(x ** 4 - 16 * x ** 2 + 5 * x, axis=1) + (40 * 10))
        return val.ravel()


class StyblinskiTang:
    """
    Styblinski-Tang Function
    See: http://www.sfu.ca/~ssurjano/stybtang.html
    """

    def __init__(self):
        self.dim = 10
        self.lb = np.full(self.dim, -5.0)
        self.ub = np.full(self.dim, 5.0)

        # optimum value(s)
        self.xopt = np.full(self.dim, -2.903534)
        self.yopt = self.__call__(self.xopt)

        # constraint
        self.cf = None

    def __call__(self, x):
        x = np.atleast_2d(x)

        term1 = x ** 4 - 16.0 * x ** 2 + 5.0 * x
        val = 0.5 * np.sum(term1, axis=1)

        return val.ravel()


if __name__ == "__main__":
    a = np.array([[0.0, 0.0],
                  [-22, 26],
                  [-8, 4],
                  [0.0, 0.0]])

    print(GRIEWANK_2().__call__(a))

    a = np.full(20, 0)
    print(GRIEWANK_20().__call__(a))

    a = np.array([[28.70220111, -2.59374183],
                  [-22.92129158, 26.23022661],
                  [-8.63085197, 4.07747505],
                  [0.0, 0.0]])
    print(ACKLEY_2().__call__(a))

    # a = np.full(2, 1)
    # print(a)
    # print(LEVY13().__call__(a))
    #
    # x = np.full(10, 1)
    # print(x)
    # print(LEVY_10.__call__(a))
