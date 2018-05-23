import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from scipy import stats


def format(floats):
    result = " "
    for v in floats:
        result += " {:f}".format(v)
    return result


def mean_interval(n, mean, s, t):
    """
    :param n - кол-во точек
    :param mean - математическое ожидани
    :param s - среднеквадратичное отклонение
    :param t - квантиль распределения Стьюдента (от (1 + Q) / 2)
    :return доверительный интервал математического ожидания для Q = 0.8
    """
    left = mean - (s / np.sqrt(n)) * t
    right = mean + (s / np.sqrt(n)) * t
    return left, right


# =============== ЧИТАЕМ ИЗ ФАЙЛА И ГОТОВИМ ДАННЫЕ ============
f = open("../input/Task_3.txt", 'r')
lines = f.readlines()
nx = int(lines[0].split(" ")[2].rstrip())
ny = int(lines[1].split(" ")[2].rstrip())

x = np.empty(nx)
# Матрица Y: nx строк, ny столбцов
y = np.empty((nx, ny))

index = 0
for xi in lines[2].split("= ")[1].rstrip().split(" "):
    x[index] = float(xi)
    index += 1

# Заполняем матрицу Y
# i - строка, j - столбец
for i in range(nx):
    line = lines[3 + i].split(" =")[1].rstrip().split(" ")
    for j in range(ny):
        y[i, j] = float(line[j])

# ================ СЧИТАЕМ СРЕДНИЕ ЗНАЧЕНИЯ ВО ВСЕХ ТОЧКАХ ====

# Матрица средних значений Yi
y_ = np.empty(nx)
# Заполняем матрицу средних значений
for i in range(nx):
    y_[i] = sum(y[i]) / ny

# Корелляционная матрица (диагональная)
sigma = np.zeros((nx, nx))
# Заполняем диагональную матрицу - дисперсий
# Так как кол-во измерений невелико, то мы учитываем только диагональные элементы
for i in range(nx):
    sum = 0
    for k in range(ny):
        sum += (y[i, k] - y_[i]) * (y[i, k] - y_[i])
    sigma[i, i] = sum / (ny - 1)

# Матрица дисперсий (диагонали sigma)
dispersions = np.empty(nx)
for i in range(nx):
    dispersions[i] = sigma[i, i]

# =============== СЧИТАЕМ ИНТЕРВАЛЬНЫЕ ОЦЕНКИ ДЛЯ СРЕДНИХ ЗНАЧЕНИЙ


Q = 0.95  # ПУСТЬ ТАК БУДЕТ - ЭТО наш доверительный интервал
# Считаем вероятность попадания в распределении Стьюдента
tinv = stats.t.ppf((1 + Q) / 2, ny - 1)
tolerant_multiplier = 3.162  # Из учебника, для n = 12 (стр 152)
confidence = np.empty((nx, 2))
tolerant = np.empty((nx, 2))
# Считаем доверительные и толерантный интервалы
for i in range(nx):
    confidence[i] = mean_interval(ny, y_[i], np.sqrt(dispersions[i]), tinv)
    tolerant[i] = [y_[i] - tolerant_multiplier * np.sqrt(dispersions[i]),
                   y_[i] + tolerant_multiplier * np.sqrt(dispersions[i])]
# Строим график средних значений, доверительного и толерантного интервалов
plt.plot(x, y_, '-r')
plt.plot(x, confidence, '--', c='grey')
plt.plot(x, tolerant, '--k')
plt.legend(('Средние значения', 'Доверительный интервал',
            'Доверительный интервал',
            'Толерантный интервал',
            'Толерантный интервал'),
           loc='lower left')
plt.savefig("../out/intervals.png", dpi=200)
plt.show()
plt.close()

# Печатаем всяикие значения
print("Значения x:")
print("\t", format(x))
print("Средние значения y:")
print("\t", format(y_))
print("Дисперсии:")
print("\t", format(dispersions))

# ========================КРИТЕРИЙ КОХНЕРА=================================

print("\nКРИТЕРИЙ КОХНЕРА")
# КоХРЕН критическое значение равно 0.07, из таблицы
kochren_krit = 0.07
kochren_stat = dispersions.max() / dispersions.sum()
# К СОЖЕЛЕНИЮ, ПРИДЕТСЯ ИСПОЛЬЗОВАТЬ ОМНК
if kochren_krit > kochren_stat:
    print("\tПоздравляю,", kochren_krit, ">", kochren_stat, ". Значит можешь использовать МНК")
else:
    print("\tСоболезную,", kochren_krit, "<", kochren_stat, ". ОМНК - твоя участь")

# print(sigma_inv)
#
# print(sigma.transpose())


# ================ АППРОКСИМАЦИЯ ПОЛИНОМАМИ И ПРОВЕРКА ИХ СТАТИСТИКИ ========
# Матрица коэффициентов для всех степеней полинома от 1 до nx - 1
# Записывать коэфф в строки будем от младших степеней, то есть от 0
a = np.zeros((nx - 2, nx))
# вектор со статистиками полиномов, степень полинома = индекс + 1
# В нем inf значит, что соответсвующий полином превысил критическое значение
approx_rating = np.empty(nx - 2)
# Зададим критерий значимости
alpha = 0.2
# В цикле перебираем все возможные степени полинома от 1 до 39 включительно
# (не до 40, иначе в формуле ny / (nx - q - 1) будет деление на ноль)

ready = False
for q in range(1, nx - 1):
    # Матрица X
    x_matrix = np.empty((nx, q + 1))
    for i in range(nx):
        for j in range(q + 1):
            x_matrix[i, j] = x[i] ** j

    # Считаем коэффициенты a_k
    # Используем эту формулу так как nx > ny + 1, и матрица Сигма диагональная
    sigma_inv = inv(sigma)
    a1 = np.matmul(x_matrix.transpose(), sigma_inv)
    a2 = inv(np.matmul(a1, x_matrix))
    a3 = np.matmul(a2, x_matrix.transpose())
    a4 = np.matmul(a3, sigma_inv)
    a_res = np.matmul(a4, y_)
    # получили вектор a_i
    # добавим его в матрицу коэффициентов
    for i in range(len(a_res)):
        a[q - 1, i] = a_res[i]
    # Заполнили матрицу коэффициентов a

    # ==============Статистика Фишера
    # Статистика
    mult = ny / (nx - q - 1)
    b1 = np.matmul(x_matrix, a_res) - y_
    b2 = np.matmul(b1.transpose(), sigma_inv)
    b3 = np.matmul(b2, b1)
    fisher_stat = mult * b3
    # Критическое значение
    # Используем эту формулу так как nx > ny + 1, и матрица Сигма диагональная (да, по то же причине)
    fisher_critical = stats.f.ppf(1 - alpha, nx - q - 1, ny - 1)
    if fisher_stat < fisher_critical:
        # Подошло
        print("\t q = ", q, '\tКрит \t', fisher_critical, "\tСтат \t", fisher_stat)
        approx_rating[q - 1] = fisher_stat
        if not ready:
            ready = True
            # Посчитаем ковариационную матрицу оценок коэффициентов только для первого полинома, который подошел
            print("\t Для q =", q)
            covar_matrix_for_a = 1 / 10 * inv(np.matmul(np.matmul(x_matrix.transpose(), sigma_inv), x_matrix))
            corel_matrix_for_a = np.empty(q + 1)
            for i in range(q + 1):
                corel_matrix_for_a[i] = covar_matrix_for_a[i, i] / (dispersions[i])
            print("\t\t Ковариационная диагональня матрица")
            print("\t\t\t", format(np.diagonal(covar_matrix_for_a)))
            print("\t\t Корелляционная диагональня матрица")
            print("\t\t\t", format(corel_matrix_for_a))
    else:
        # Не подходит
        approx_rating[q - 1] = np.inf

print(approx_rating)

# Посчитаем ковариационную матрицу оценок коэффициентов для полинома с наименьшей степенью и
# с наименьшей статистикой, но такие, которые прошли отбор
# ТУТ ХРАНЯТСЯ ИМЕННО ИНДЕСЫ, А НЕ СТЕПЕНЬ ПОЛИНОМА
min_power_index = 0
for i in range(len(approx_rating)):
    if approx_rating[i] != np.inf:
        min_power_index = i
        break
min_stat_index = 0
for i in range(len(approx_rating)):
    if approx_rating[i] < approx_rating[min_stat_index]:
        min_stat_index = i

print(min_power_index)
print(min_stat_index)


# ============= ПОСТРОИМ ГРАФИКИ ДЛЯ ПОЛИНОМОВ И CРЕДНИХ ЗНАЧЕНИЙ

def polinom(coefficients, x):
    sum = 0
    for i in range(len(coefficients)):
        sum += coefficients[i] * (x ** i)
    return sum


# Строим графики
plt.plot(x, y_)
plt.plot(x, polinom(a[min_stat_index], x), '--')
plt.plot(x, polinom(a[min_power_index], x), '-.')
plt.plot(x, polinom(a[0], x), ':', c='grey')
text_1 = 'Полином степени ' + str(min_stat_index + 1)
text_2 = 'Полином степени ' + str(min_power_index + 1)
text_3 = 'Полином степени 1'
plt.legend(('Средние значения', text_1,
            text_2, text_3), loc='lower left')
plt.savefig("../out/result_approximations.png", dpi=200)
plt.show()
plt.close()
