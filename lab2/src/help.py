from math import sqrt
import numpy as np
import scipy.stats as stats
import scipy.special as special


def central_moment(data, k, mean):
    """
    :param data - данные
    :param k - порядок
    :param mean - математическое ожидание
    :return центральной момент k-го порядка
    """
    sum_ = 0
    for value in data:
        sum_ += (value - mean) ** k
    return sum_ / (len(data))


def dispersion(data, mean):
    """
    :param data - данные
    :param mean - математическое ожидание
    :return дисперсия
    """
    sum_ = 0
    for value in data:
        sum_ += (value - mean) ** 2
    return sum_ / (len(data) - 1)


def quantile(n, p):
    """
    :param n - кол-во точек
    :param p - вероятность
    :return квантиль
    """
    return int(n * p)


def interquantile_interval(n, p):
    """
    :param n - кол-во точек
    :param p - вероятность
    :return интерквантильный промежуток
    """
    result = (quantile(n, (1 - p) / 2), quantile(n, (1 + p) / 2))
    return result


def kurtosis(fourth_central_moment, dispersion):
    """
    :param fourth_central_moment - четвертый центральный момент
    :param dispersion - дисперсия
    :return эксцесса
    """
    return fourth_central_moment / dispersion ** 2


def asymmetry(third_central_moment, dispersion):
    """
    :param third_central_moment - третий центральный момент
    :param dispersion - дисперсия
    :return коэффициент асимметрии
    """
    return third_central_moment / sqrt(dispersion ** 3)


def mean_interval(n, mean, s, t):
    """
    :param n - кол-во точек
    :param mean - математическое ожидани
    :param s - среднеквадратичное отклонение
    :param t - квантиль распределения Стьюдента (от (1 + Q) / 2)
    :return доверительный интервал математического ожидания для Q = 0.8
    """
    left = mean - (s / sqrt(n)) * t
    right = mean + (s / sqrt(n)) * t
    return left, right


def dispersion_interval(n, dispersion, left_chi2inv, right_chi2inv):
    """
    :param n - кол-во точек
    :param dispersion - дисперсия
    :param left_chi2inv - квантиль распределения хи-квадрат (от (1 + Q) / 2)
    :param right_chi2inv - квантиль распределения хи-квадрат (от (1 - Q) / 2)
    :return доверительный интервал дисперсии для Q = 0.8
    """
    left = dispersion * (n - 1) / left_chi2inv
    right = dispersion * (n - 1) / right_chi2inv
    return left, right


def find_k(n, p, q):
    """
    Используем функцию cdf из библиотеки scipy.stats
    Завершаем цикл, как только находим такое к при
    котором будет выполняться f(x) <= 1 - q

    :param n - размер выборки
    :param p - вероятность
    :param q - доверительная вероятность
    :return: кол-во точек, которые нужно отбросить
    """
    for k in range(n):
        if stats.binom.cdf(n - k, n, p) <= 1 - q: return k


class gammaGradient:
    c: float

    def __init__(self, c):
        self.c = c

    def gamma_gradient(self, k):
        """
        :param k: значение k Гамма-функции
        :return: значение
        """
        return np.log(k) - special.digamma(k) - self.c


def fmin_bisection(func, a: float, b: float, e: float):
    result = 0
    while abs(a - b) > e:
        c = (a + b) / 2
        left = func(a)
        center = func(c)
        if np.sign(left) == np.sign(center):
            result = a = c
        else:
            result = b = c
    return result
