import numpy as np
import math
import csv
import numpy as np
import random
import copy  # array-copying convenience
import sys  # max float
import pandas as pd

class CEC_functions:
    def __init__(self, dim):
        csv_file = open('extdata/M_D' + str(dim) + '.txt')
        csv_data = csv.reader(csv_file, delimiter=' ')
        csv_data_not_null = [[float(data) for data in row if len(data) > 0] for row in csv_data]
        self.rotate_data = np.array(csv_data_not_null)
        csv_file = open('extdata/shift_data.txt')
        csv_data = csv.reader(csv_file, delimiter=' ')
        self.sd = []
        for row in csv_data:
            self.sd += [float(data) for data in row if len(data) > 0]
        self.M1 = self.read_M(dim, 0)
        self.M2 = self.read_M(dim, 1)
        self.O = self.shift_data(dim, 0)
        self.aux9_1 = np.array([0.5 ** j for j in range(0, 21)])
        self.aux9_2 = np.array([3 ** j for j in range(0, 21)])
        self.aux16 = np.array([2 ** j for j in range(1, 33)])

    def read_M(self, dim, m):
        return self.rotate_data[m * dim: (m + 1) * dim]

    def shift_data(self, dim, m):
        return np.array(self.sd[m * dim: (m + 1) * dim])

    def carat(self, dim, alpha):  # I don't know this is correct or not!!!
        return alpha ** (np.arange(dim) / (2 * (dim - 1)))

    def T_asy(self, X, Y, beta):
        D = len(X)
        for i in range(D):
            if X[i] > 0:
                Y[i] = X[i] ** (1 + beta * (i / (D - 1)) * np.sqrt(X[i]))
        pass

    def T_osz(self, X):
        for i in [0, -1]:
            c1 = 10 if X[i] > 0 else 5.5
            c2 = 7.9 if X[i] > 0 else 3.1
            x_hat = 0 if X[i] == 0 else np.log(abs(X[i]))
            X[i] = np.sign(X[i]) * np.exp(x_hat + 0.049 * (np.sin(c1 * x_hat) + np.sin(c2 * x_hat)))
        pass

    def cf_cal(self, X, delta, bias, fit):
        d = len(X)
        W_star = []
        cf_num = len(fit)

        for i in range(cf_num):
            X_shift = X - self.shift_data(d, i)
            W = 1 / np.sqrt(np.sum(X_shift ** 2)) * np.exp(-1 * np.sum(X_shift ** 2) / (2 * d * delta[i] ** 2))
            W_star.append(W)

        if (np.max(W_star) == 0):
            W_star = [1] * cf_num

        omega = W_star / np.sum(W_star) * (fit + bias)

        return np.sum(omega)

    def Y(self, X, fun_num, rflag=None):
        if rflag is None:
            rf = [0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1][fun_num - 1]
        else:
            rf = rflag

        # Unimodal Functions
        # Sphere Function
        # 1
        if fun_num == 1:
            Z = X - self.O
            if rf == 1:
                Z = self.M1 @ Z
            Y = np.sum(Z ** 2) - 1400


        # Rotated High Conditioned Elliptic Function
        # 2
        elif fun_num == 2:
            d = len(X)
            X_shift = X - self.O
            X_rotate = self.M1 @ X_shift
            self.T_osz(X_rotate)
            Y = np.sum((1e6 ** (np.arange(d) / (d - 1))) * X_rotate ** 2) - 1300


        # Rotated Bent Cigar Function
        # 3
        elif fun_num == 3:
            X_shift = X - self.O
            X_rotate = self.M1 @ X_shift
            self.T_asy(X_rotate, X_shift, 0.5)
            Z = self.M2 @ X_shift

            Y = Z[0] ** 2 + 1e6 * np.sum(Z[1:] ** 2) - 1200


        # Rotated Discus Function
        # 4
        elif fun_num == 4:
            d = len(X)
            X_shift = X - self.O
            X_rotate = self.M1 @ X_shift
            self.T_osz(X_rotate)
            Y = (1e6) * (X_rotate[0] ** 2) + np.sum(X_rotate[1:d] ** 2) - 1100


        # Different Powers Function
        # 5
        elif fun_num == 5:
            d = len(X)
            Z = X - self.O
            if rf == 1:
                Z = self.M1 @ Z
            Y = np.sqrt(np.sum(abs(Z) ** (2 + (4 * np.arange(d) / (d - 1)).astype(int)))) - 1000

        # Basic Multimodal Functions
        # Rotated Rosenbrock’s Function
        # 6
        elif fun_num == 6:
            d = len(X)
            X_shift = X - self.O
            X_rotate = self.M1 @ ((2.048 * X_shift) / 100)
            Z = X_rotate + 1
            Y = np.sum(100 * (Z[:d - 1] ** 2 - Z[1:d]) ** 2 + (Z[:d - 1] - 1) ** 2) - 900


        # Rotated Schaffers F7 Function
        # 7
        elif fun_num == 7:
            d = len(X)
            X_shift = X - self.O
            X_rotate = self.M1 @ X_shift
            self.T_asy(X_rotate, X_shift, 0.5)
            Z = self.M2 @ (self.carat(d, 10) * X_shift)

            Z = np.sqrt(Z[:-1] ** 2 + Z[1:] ** 2)
            Y = (np.sum(np.sqrt(Z) + np.sqrt(Z) * np.sin(50 * Z ** 0.2) ** 2) / (d - 1)) ** 2 - 800

        # Rotated Ackley’s Function
        # 8
        elif fun_num == 8:
            d = len(X)
            X_shift = X - self.O
            X_rotate = self.M1 @ X_shift
            self.T_asy(X_rotate, X_shift, 0.5)
            Z = self.M2 @ (self.carat(d, 10) * X_shift)

            Y = -20 * np.exp(-0.2 * np.sqrt(np.sum(Z ** 2) / d)) \
                - np.exp(np.sum(np.cos(2 * np.pi * Z)) / d) \
                + 20 + np.exp(1) - 700


        # Rotated Weierstrass Function
        # 9
        elif fun_num == 9:
            d = len(X)
            X_shift = 0.005 * (X - self.O)
            X_rotate_1 = self.M1 @ (X_shift)
            self.T_asy(X_rotate_1, X_shift, 0.5)
            Z = self.M2 @ (self.carat(d, 10) * X_shift)

            # kmax = 20

            def _w(v):
                return np.sum(self.aux9_1 * np.cos(2.0 * np.pi * self.aux9_2 * v))

            Y = np.sum([_w(Z[i] + 0.5) for i in range(d)]) - (_w(0.5) * d) - 600


        # Rotated Griewank’s Function
        # 10
        elif fun_num == 10:
            d = len(X)
            X_shift = (600.0 * (X - self.O)) / 100.0
            X_rotate = self.M1 @ X_shift
            Z = self.carat(d, 100) * X_rotate  # used carat as matrix not sure though!!!

            Y = 1.0 + (np.sum(Z ** 2) / 4000.0) - np.multiply.reduce(np.cos(Z / np.sqrt(np.arange(1, d + 1)))) - 500


        # Rastrigin’s Function
        # 11
        elif fun_num == 11:
            d = len(X)
            X_shift = 0.0512 * (X - self.O)
            if rf == 1:
                X_shift = self.M1 @ X_shift
            X_osz = X_shift.copy()
            self.T_osz(X_osz)
            self.T_asy(X_osz, X_shift, 0.2)
            if rf == 1:
                X_shift = self.M2 @ X_shift
            Z = self.carat(d, 10) * X_shift

            Y = np.sum(10 + Z ** 2 - 10 * np.cos(2 * np.pi * Z)) - 400

        # Rotated Rastrigin’s Function
        # 12
        elif fun_num == 12:
            d = len(X)
            X_shift = 0.0512 * (X - self.O)
            X_rotate = self.M1 @ X_shift
            X_hat = X_rotate.copy()
            self.T_osz(X_hat)
            self.T_asy(X_hat, X_rotate, 0.2)
            X_rotate = self.M2 @ X_rotate
            X_rotate = self.carat(d, 10) * X_rotate
            Z = self.M1 @ X_rotate

            Y = np.sum(10 + Z ** 2 - 10 * np.cos(2 * np.pi * Z)) - 300


        # Non-continuous Rotated Rastrigin’s Function
        # 13
        elif fun_num == 13:
            d = len(X)
            X_shift = 0.0512 * (X - self.O)
            X_hat = self.M1 @ X_shift

            X_hat[abs(X_hat) > 0.5] = np.round(X_hat[abs(X_hat) > 0.5] * 2) / 2
            Y_hat = X_hat.copy()
            self.T_osz(X_hat)
            self.T_asy(X_hat, Y_hat, 0.2)
            X_rotate = self.M2 @ Y_hat
            X_carat = self.carat(d, 10) * X_rotate
            Z = self.M1 @ X_carat

            Y = np.sum(10 + Z ** 2 - 10 * np.cos(2 * np.pi * Z)) - 200

        # Schwefel’s Function
        # 14 15
        elif fun_num == 14 or fun_num == 15:
            d = len(X)
            X_shift = 10 * (X - self.O)
            if rf:
                X_shift = self.M1 @ X_shift
            Z = self.carat(d, 10) * X_shift + 420.9687462275036
            Z[abs(Z) <= 500] = \
                Z[abs(Z) <= 500] * np.sin(np.sqrt(abs(Z[abs(Z) <= 500])))
            Z[Z > 500] = \
                (500 - Z[Z > 500] % 500) * np.sin(np.sqrt(500 - Z[Z > 500] % 500)) \
                - (Z[Z > 500] - 500) ** 2 / (10000 * d)
            Z[Z < -500] = \
                (abs(Z[Z < -500]) % 500 - 500) * np.sin(np.sqrt(500 - abs(Z[Z < -500]) % 500)) \
                - (Z[Z < -500] + 500) ** 2 / (10000 * d)
            Y = 418.9828872724338 * d - np.sum(Z) + (-100 if fun_num == 14 else 100)

        # Rotated Katsuura Function
        # 16
        elif fun_num == 16:
            d = len(X)
            X_shift = 0.05 * (X - self.O)
            X_rotate = self.M1 @ X_shift
            X_carat = self.carat(d, 100) * X_rotate
            Z = self.M2 @ X_carat

            def _kat(c):
                return np.sum(np.abs(self.aux16 * c - np.round(self.aux16 * c)) / self.aux16)

            for i in range(d):
                Z[i] = (1 + (i + 1) * _kat(Z[i]))

            Z = np.multiply.reduce(Z ** (10 / d ** 1.2))
            Y = (10 / d ** 2) * Z - (10 / d ** 2) + 200


        # bi-Rastrigin Function
        # 17 18
        elif fun_num == 17 or fun_num == 18:
            d = len(X)
            mu_0 = 2.5
            S = 1 - 1 / ((2 * np.sqrt(d + 20)) - 8.2)
            mu_1 = -1 * np.sqrt((mu_0 ** 2 - 1) / S)
            X_star = self.O
            X_shift = 0.1 * (X - self.O)
            X_hat = []
            for i in range(d):
                X_hat.append(2 * np.sign(X_star[i]) * X_shift[i] + mu_0)

            MU_0 = np.ones(dim) * mu_0
            Z = X_hat - MU_0
            if rf:
                Z = self.M1 @ Z
            Z = self.carat(d, 100) * Z
            if rf:
                Z = self.M2 @ Z

            Y_1 = []
            Y_2 = []
            for i in range(d):
                Y_1.append((X_hat[i] - mu_0) ** 2)
                Y_2.append((X_hat[i] - mu_1) ** 2)

            Y_3 = np.minimum(np.sum(Y_1), d + S * np.sum(Y_2))
            Y = Y_3 + 10 * (d - np.sum(np.cos(2 * np.pi * Z))) + (300 if fun_num == 17 else 400)

        # Rotated Expanded Griewank’s plus Rosenbrock’s Function
        # 19
        elif fun_num == 19:
            d = len(X)
            X_shift = 0.05 * (X - self.O) + 1

            tmp = X_shift ** 2 - np.roll(X_shift, -1)
            tmp = 100 * tmp ** 2 + (X_shift - 1) ** 2
            Z = np.sum(tmp ** 2 / 4000 - np.cos(tmp) + 1)

            Y = Z + 500

        # Rotated Expanded Scaffer’s F6 Function
        # 20
        elif fun_num == 20:
            d = len(X)
            X_shift = X - self.O
            X_rotate = self.M1 @ X_shift
            self.T_asy(X_rotate, X_shift, 0.5)
            Z = self.M2 @ X_shift

            tmp1 = Z ** 2 + (np.roll(Z, -1)) ** 2

            Y = np.sum(0.5 + (np.sin(np.sqrt(tmp1)) ** 2 - 0.5) / (1 + 0.001 * tmp1) ** 2) + 600

        # New Composition Functions
        # Composition Function 1
        # 21
        elif fun_num == 21:
            d = len(X)
            delta = np.array([10, 20, 30, 40, 50])
            bias = np.array([0, 100, 200, 300, 400])
            fit = []
            fit.append((self.Y(X, 6, rf) + 900) / 1)
            self.O = self.shift_data(d, 1)
            self.M1 = self.read_M(d, 1)
            self.M2 = self.read_M(d, 2)
            fit.append((self.Y(X, 5, rf) + 1000) / 1e6)
            self.O = self.shift_data(d, 2)
            self.M1 = self.read_M(d, 2)
            self.M2 = self.read_M(d, 3)
            fit.append((self.Y(X, 3, rf) + 1200) / 1e26)
            self.O = self.shift_data(d, 3)
            self.M1 = self.read_M(d, 3)
            self.M2 = self.read_M(d, 4)
            fit.append((self.Y(X, 4, rf) + 1100) / 1e6)
            self.O = self.shift_data(d, 4)
            self.M1 = self.read_M(d, 4)
            self.M2 = self.read_M(d, 5)
            fit.append((self.Y(X, 1, rf) + 1400) / 1e1)

            Y = self.cf_cal(X, delta, bias, np.array(fit)) + 700

        elif fun_num == 22:
            d = len(X)
            delta = np.array([20, 20, 20])
            bias = np.array([0, 100, 200])
            fit = []
            fit.append((self.Y(X, 14, rf) + 100) / 1)
            self.O = self.shift_data(d, 1)
            self.M1 = self.read_M(d, 1)
            self.M2 = self.read_M(d, 2)
            fit.append((self.Y(X, 14, rf) + 100) / 1)
            self.O = self.shift_data(d, 2)
            self.M1 = self.read_M(d, 2)
            self.M2 = self.read_M(d, 3)
            fit.append((self.Y(X, 14, rf) + 100) / 1)

            Y = Y = self.cf_cal(X, delta, bias, np.array(fit)) + 800

        elif fun_num == 23:
            d = len(X)
            delta = np.array([20, 20, 20])
            bias = np.array([0, 100, 200])
            fit = []
            fit.append((self.Y(X, 15, rf) - 100) / 1)
            self.O = self.shift_data(d, 1)
            self.M1 = self.read_M(d, 1)
            self.M2 = self.read_M(d, 2)
            fit.append((self.Y(X, 15, rf) - 100) / 1)
            self.O = self.shift_data(d, 2)
            self.M1 = self.read_M(d, 2)
            self.M2 = self.read_M(d, 3)
            fit.append((self.Y(X, 15, rf) - 100) / 1)

            Y = self.cf_cal(X, delta, bias, np.array(fit)) + 900

        elif fun_num == 24:
            d = len(X)
            delta = np.array([20, 20, 20])
            bias = np.array([0, 100, 200])
            fit = []
            fit.append((self.Y(X, 15, rf) - 100) * 0.25)
            self.O = self.shift_data(d, 1)
            self.M1 = self.read_M(d, 1)
            self.M2 = self.read_M(d, 2)
            fit.append((self.Y(X, 12, rf) + 300) * 1)
            self.O = self.shift_data(d, 2)
            self.M1 = self.read_M(d, 2)
            self.M2 = self.read_M(d, 3)
            fit.append((self.Y(X, 9, rf) + 600) * 2.5)

            Y = self.cf_cal(X, delta, bias, np.array(fit)) + 1000

        elif fun_num == 25:
            d = len(X)
            delta = np.array([10, 30, 50])
            bias = np.array([0, 100, 200])
            fit = []
            fit.append((self.Y(X, 15, rf) - 100) * 0.25)
            self.O = self.shift_data(d, 1)
            self.M1 = self.read_M(d, 1)
            self.M2 = self.read_M(d, 2)
            fit.append((self.Y(X, 12, rf) + 300) * 1)
            self.O = self.shift_data(d, 2)
            self.M1 = self.read_M(d, 2)
            self.M2 = self.read_M(d, 3)
            fit.append((self.Y(X, 9, rf) + 600) * 2.5)

            Y = self.cf_cal(X, delta, bias, np.array(fit)) + 1100

        elif fun_num == 26:
            d = len(X)
            delta = np.array([10, 10, 10, 10, 10])
            bias = np.array([0, 100, 200, 300, 400])
            fit = []
            fit.append((self.Y(X, 15, rf) - 100) * 0.25)
            self.O = self.shift_data(d, 1)
            self.M1 = self.read_M(d, 1)
            self.M2 = self.read_M(d, 2)
            fit.append((self.Y(X, 12, rf) + 300) * 1)
            self.O = self.shift_data(d, 2)
            self.M1 = self.read_M(d, 2)
            self.M2 = self.read_M(d, 3)
            fit.append((self.Y(X, 2, rf) + 1300) / 1e7)
            self.O = self.shift_data(d, 3)
            self.M1 = self.read_M(d, 3)
            self.M2 = self.read_M(d, 4)
            fit.append((self.Y(X, 9, rf) + 600) * 2.5)
            self.O = self.shift_data(d, 4)
            self.M1 = self.read_M(d, 4)
            self.M2 = self.read_M(d, 5)
            fit.append((self.Y(X, 10, rf) + 500) * 10)

            Y = self.cf_cal(X, delta, bias, np.array(fit)) + 1200
        elif fun_num == 27:
            d = len(X)
            delta = np.array([10, 10, 10, 20, 20])
            bias = np.array([0, 100, 200, 300, 400])
            fit = []
            fit.append((self.Y(X, 10, rf) + 500) * 100)
            self.O = self.shift_data(d, 1)
            self.M1 = self.read_M(d, 1)
            self.M2 = self.read_M(d, 2)
            fit.append((self.Y(X, 12, rf) + 300) * 10)
            self.O = self.shift_data(d, 2)
            self.M1 = self.read_M(d, 2)
            self.M2 = self.read_M(d, 3)
            fit.append((self.Y(X, 15, rf) - 100) * 2.5)
            self.O = self.shift_data(d, 3)
            self.M1 = self.read_M(d, 3)
            self.M2 = self.read_M(d, 4)
            fit.append((self.Y(X, 9, rf) + 600) * 25)
            self.O = self.shift_data(d, 4)
            self.M1 = self.read_M(d, 4)
            self.M2 = self.read_M(d, 5)
            fit.append((self.Y(X, 1, rf) + 1400) / 1e1)

            Y = self.cf_cal(X, delta, bias, np.array(fit)) + 1300

        elif fun_num == 28:
            d = len(X)
            delta = np.array([10, 20, 30, 40, 50])
            bias = np.array([0, 100, 200, 300, 400])
            fit = []
            fit.append((self.Y(X, 19, rf) - 500) * 2.5)
            self.O = self.shift_data(d, 1)
            self.M1 = self.read_M(d, 1)
            self.M2 = self.read_M(d, 2)
            fit.append(((self.Y(X, 7, rf) + 800) * 2.5) / 1e3)
            self.O = self.shift_data(d, 2)
            self.M1 = self.read_M(d, 2)
            self.M2 = self.read_M(d, 3)
            fit.append((self.Y(X, 15, rf) - 100) * 2.5)
            self.O = self.shift_data(d, 3)
            self.M1 = self.read_M(d, 3)
            self.M2 = self.read_M(d, 4)
            fit.append(((self.Y(X, 20, rf) - 600) * 5) / 1e4)
            self.O = self.shift_data(d, 4)
            self.M1 = self.read_M(d, 4)
            self.M2 = self.read_M(d, 5)
            fit.append((self.Y(X, 1, rf) + 1400) / 1e1)

            Y = self.cf_cal(X, delta, bias, np.array(fit)) + 1400

        return Y

dim=50
fitness = CEC_functions(dim)
# wolf class
class wolf:
    def __init__(self, fnum, minx, maxx, seed):
        self.rnd = random.Random(seed)
        self.position = [0.0 for i in range(dim)]

        for i in range(dim):
            self.position[i] = ((maxx - minx) * self.rnd.random() + minx)
        self.fitness = fitness.Y(self.position, fnum)  # curr fitness


# grey wolf optimization (GWO)
def gwo(fnum, max_iter, n, minx, maxx):


    rnd = random.Random(0)

    # create n random wolves
    population = [wolf(fnum, minx, maxx, i) for i in range(n)]

    # On the basis of fitness values of wolves
    # sort the population in asc order
    population = sorted(population, key=lambda temp: temp.fitness)

    # best 3 solutions will be called as
    # alpha, beta and gama
    alpha_wolf, beta_wolf, gamma_wolf = copy.copy(population[: 3])

    # main loop of gwo
    Iter = 0
    while Iter < max_iter:

        # after every 10 iterations
        # print iteration number and best fitness value so far
        #if Iter % 10 == 0 and Iter > 1:
            #print("Iter = " + str(Iter) + " best fitness = %.3f" % alpha_wolf.fitness)

        # linearly decreased from 2 to 0
        a = 2 * (1 - Iter / max_iter)

        # updating each population member with the help of best three members
        for i in range(n):
            A1, A2, A3 = a * (2 * rnd.random() - 1), a * (
                    2 * rnd.random() - 1), a * (2 * rnd.random() - 1)
            C1, C2, C3 = 2 * rnd.random(), 2 * rnd.random(), 2 * rnd.random()

            X1 = [0.0 for x in range(dim)]
            X2 = [0.0 for x in range(dim)]
            X3 = [0.0 for x in range(dim)]
            Xnew = [0.0 for x in range(dim)]
            for j in range(dim):
                X1[j] = alpha_wolf.position[j] - A1 * abs(
                    C1 * alpha_wolf.position[j] - population[i].position[j])
                X2[j] = beta_wolf.position[j] - A2 * abs(
                    C2 * beta_wolf.position[j] - population[i].position[j])
                X3[j] = gamma_wolf.position[j] - A3 * abs(
                    C3 * gamma_wolf.position[j] - population[i].position[j])
                Xnew[j] = X1[j] + X2[j] + X3[j]

            for j in range(dim):
                Xnew[j] /= 3.0

            # fitness calculation of new solution
            fnew = fitness.Y(Xnew,fnum)

            # greedy selection
            if fnew < population[i].fitness:
                population[i].position = Xnew
                population[i].fitness = fnew

        # On the basis of fitness values of wolves
        # sort the population in asc order
        population = sorted(population, key=lambda temp: temp.fitness)

        # best 3 solutions will be called as
        # alpha, beta and gama
        alpha_wolf, beta_wolf, gamma_wolf = copy.copy(population[: 3])

        Iter += 1
    # end-while

    # returning the best solution
    return alpha_wolf.position

# improved grey wolf optimization (IGWO) Long 2017 =====>
def igwo(fnum, max_iter, n, minx, maxx):
    rnd = random.Random(0)

    # create n random wolves
    population = [wolf(fnum, minx, maxx, i) for i in range(n)]

    # On the basis of fitness values of wolves
    # sort the population in asc order
    population = sorted(population, key=lambda temp: temp.fitness)

    # best 3 solutions will be called as
    # alpha, beta and gama
    alpha_wolf, beta_wolf, gamma_wolf = copy.copy(population[: 3])

    # main loop of gwo
    Iter = 0
    while Iter < max_iter:

        # after every 10 iterations
        # print iteration number and best fitness value so far
        #if Iter % 10 == 0 and Iter > 1:
            #print("Iter = " + str(Iter) + " best fitness = %.3f" % alpha_wolf.fitness)

        # linearly decreased from 2 to 0
        #a = 2 * (1 - Iter / max_iter)
        a = 2 - 2 * math.tan((math.pi * Iter) / (5 * max_iter))
        # updating each population member with the help of best three members
        for i in range(n):
            A1, A2, A3 = a * (2 * rnd.random() - 1), a * (
                    2 * rnd.random() - 1), a * (2 * rnd.random() - 1)
            C1, C2, C3 = 2 * rnd.random(), 2 * rnd.random(), 2 * rnd.random()

            X1 = [0.0 for x in range(dim)]
            X2 = [0.0 for x in range(dim)]
            X3 = [0.0 for x in range(dim)]
            Xnew = [0.0 for x in range(dim)]
            for j in range(dim):
                X1[j] = alpha_wolf.position[j] - A1 * abs(
                    C1 * alpha_wolf.position[j] - population[i].position[j])
                X2[j] = beta_wolf.position[j] - A2 * abs(
                    C2 * beta_wolf.position[j] - population[i].position[j])
                X3[j] = gamma_wolf.position[j] - A3 * abs(
                    C3 * gamma_wolf.position[j] - population[i].position[j])
                Xnew[j] = X1[j] + X2[j] + X3[j]

            for j in range(dim):
                Xnew[j] /= 3.0

            # fitness calculation of new solution
            fnew = fitness.Y(Xnew, fnum)

            # greedy selection
            if fnew < population[i].fitness:
                population[i].position = Xnew
                population[i].fitness = fnew

        # On the basis of fitness values of wolves
        # sort the population in asc order
        population = sorted(population, key=lambda temp: temp.fitness)

        # best 3 solutions will be called as
        # alpha, beta and gama
        alpha_wolf, beta_wolf, gamma_wolf = copy.copy(population[: 3])

        Iter += 1
    # end-while

    # returning the best solution
    return alpha_wolf.position

# Logarithimc factor  grey wolf optimization (LFGWO) T Wu 2018 ====>
def lfgwo(fnum, max_iter, n, minx, maxx):
    rnd = random.Random(0)

    # create n random wolves
    population = [wolf(fnum, minx, maxx, i) for i in range(n)]

    # On the basis of fitness values of wolves
    # sort the population in asc order
    population = sorted(population, key=lambda temp: temp.fitness)

    # best 3 solutions will be called as
    # alpha, beta and gama
    alpha_wolf, beta_wolf, gamma_wolf = copy.copy(population[: 3])

    # main loop of gwo
    Iter = 0
    while Iter < max_iter:

        # after every 10 iterations
        # print iteration number and best fitness value so far
        #if Iter % 10 == 0 and Iter > 1:
            #print("Iter = " + str(Iter) + " best fitness = %.3f" % alpha_wolf.fitness)

        # linearly decreased from 2 to 0
        #a = 2 * (1 - Iter / max_iter)
        #a = 2 - 2 * math.tan((math.pi * Iter) / (5 * max_iter))
        a = 2 - math.log10(1 + (6 * 0.8) * (Iter / max_iter))
        # updating each population member with the help of best three members
        for i in range(n):
            A1, A2, A3 = a * (2 * rnd.random() - 1), a * (
                    2 * rnd.random() - 1), a * (2 * rnd.random() - 1)
            C1, C2, C3 = 2 * rnd.random(), 2 * rnd.random(), 2 * rnd.random()

            X1 = [0.0 for x in range(dim)]
            X2 = [0.0 for x in range(dim)]
            X3 = [0.0 for x in range(dim)]
            Xnew = [0.0 for x in range(dim)]

            for j in range(dim):
                X1[j] = alpha_wolf.position[j] - A1 * abs(
                    C1 * alpha_wolf.position[j] - population[i].position[j])
                X2[j] = beta_wolf.position[j] - A2 * abs(
                    C2 * beta_wolf.position[j] - population[i].position[j])
                X3[j] = gamma_wolf.position[j] - A3 * abs(
                    C3 * gamma_wolf.position[j] - population[i].position[j])
                Xnew[j] = X1[j] + X2[j] + X3[j]

            for j in range(dim):
                Xnew[j] /= 3.0

            # fitness calculation of new solution
            fnew = fitness.Y(Xnew, fnum)

            # greedy selection
            if fnew < population[i].fitness:
                population[i].position = Xnew
                population[i].fitness = fnew

        # On the basis of fitness values of wolves
        # sort the population in asc order
        population = sorted(population, key=lambda temp: temp.fitness)

        # best 3 solutions will be called as
        # alpha, beta and gama
        alpha_wolf, beta_wolf, gamma_wolf = copy.copy(population[: 3])

        Iter += 1
    # end-while

    # returning the best solution
    return alpha_wolf.position

# nonlinear  grey wolf optimization (NGGWO) M Wang 2016 =====>
def ngwo(fnum, max_iter, n, minx, maxx):
    rnd = random.Random(0)

    # create n random wolves
    population = [wolf(fnum, minx, maxx, i) for i in range(n)]

    # On the basis of fitness values of wolves
    # sort the population in asc order
    population = sorted(population, key=lambda temp: temp.fitness)

    # best 3 solutions will be called as
    # alpha, beta and gama
    alpha_wolf, beta_wolf, gamma_wolf = copy.copy(population[: 3])

    # main loop of gwo
    Iter = 0
    while Iter < max_iter:

        # after every 10 iterations
        # print iteration number and best fitness value so far
        #if Iter % 10 == 0 and Iter > 1:
            #print("Iter = " + str(Iter) + " best fitness = %.3f" % alpha_wolf.fitness)

        # linearly decreased from 2 to 0
        #a = 2 * (1 - Iter / max_iter) #GWO
        #a = 2 - 2 * math.tan((math.pi * Iter) / (5 * max_iter)) # IGWO
        #a = 2 - math.log10(1 + (6 * 0.8) * (Iter / max_iter))  # LFGWO
        a = 2-2*((1-Iter/max_iter)**2)   #NGWO

        # updating each population member with the help of best three members
        for i in range(n):
            A1, A2, A3 = a * (2 * rnd.random() - 1), a * (
                    2 * rnd.random() - 1), a * (2 * rnd.random() - 1)
            C1, C2, C3 = 2 * rnd.random(), 2 * rnd.random(), 2 * rnd.random()

            X1 = [0.0 for x in range(dim)]
            X2 = [0.0 for x in range(dim)]
            X3 = [0.0 for x in range(dim)]
            Xnew = [0.0 for x in range(dim)]

            for j in range(dim):
                X1[j] = alpha_wolf.position[j] - A1 * abs(
                    C1 * alpha_wolf.position[j] - population[i].position[j])
                X2[j] = beta_wolf.position[j] - A2 * abs(
                    C2 * beta_wolf.position[j] - population[i].position[j])
                X3[j] = gamma_wolf.position[j] - A3 * abs(
                    C3 * gamma_wolf.position[j] - population[i].position[j])
                Xnew[j] = X1[j] + X2[j] + X3[j]

            for j in range(dim):
                Xnew[j] /= 3.0

            # fitness calculation of new solution
            fnew = fitness.Y(Xnew, fnum)

            # greedy selection
            if fnew < population[i].fitness:
                population[i].position = Xnew
                population[i].fitness = fnew

        # On the basis of fitness values of wolves
        # sort the population in asc order
        population = sorted(population, key=lambda temp: temp.fitness)

        # best 3 solutions will be called as
        # alpha, beta and gama
        alpha_wolf, beta_wolf, gamma_wolf = copy.copy(population[: 3])

        Iter += 1
    # end-while

    # returning the best solution
    return alpha_wolf.position

# variable weight  grey wolf optimization (VMGWO) Kumar 2020 ====>
def vwgwo(fnum, max_iter, n, minx, maxx):
    rnd = random.Random(0)

    # create n random wolves
    population = [wolf(fnum, minx, maxx, i) for i in range(n)]

    # On the basis of fitness values of wolves
    # sort the population in asc order
    population = sorted(population, key=lambda temp: temp.fitness)

    # best 3 solutions will be called as
    # alpha, beta and gama
    alpha_wolf, beta_wolf, gamma_wolf = copy.copy(population[: 3])

    # main loop of gwo
    Iter = 0
    while Iter < max_iter:

        # after every 10 iterations
        # print iteration number and best fitness value so far
        # if Iter % 10 == 0 and Iter > 1:
        # print("Iter = " + str(Iter) + " best fitness = %.3f" % alpha_wolf.fitness)

        # linearly decreased from 2 to 0
        a = 2 * (1 - Iter / max_iter) #GWO / VMGWO
        # a = 2 - 2 * math.tan((math.pi * Iter) / (5 * max_iter)) # IGWO
        # a = 2 - math.log10(1 + (6 * 0.8) * (Iter / max_iter))  # LFGWO
        #a = 2 - 2 * ((1 - Iter / max_iter) ** 2)  # NGWO
        #a=2/max_iter*math.sqrt(max_iter**2-(Iter+1)**2) #EGWO

        # updating each population member with the help of best three members
        for i in range(n):
            A1, A2, A3 = a * (2 * rnd.random() - 1), a * (
                    2 * rnd.random() - 1), a * (2 * rnd.random() - 1)
            C1, C2, C3 = 2 * rnd.random(), 2 * rnd.random(), 2 * rnd.random()

            X1 = [0.0 for x in range(dim)]
            X2 = [0.0 for x in range(dim)]
            X3 = [0.0 for x in range(dim)]
            Xnew = [0.0 for x in range(dim)]
            ac = 0.5
            bc = 0.3
            gc = 0.2
            for j in range(dim):
                X1[j] = alpha_wolf.position[j] - A1 * abs(
                    C1 * alpha_wolf.position[j] - population[i].position[j])
                X2[j] = beta_wolf.position[j] - A2 * abs(
                    C2 * beta_wolf.position[j] - population[i].position[j])
                X3[j] = gamma_wolf.position[j] - A3 * abs(
                    C3 * gamma_wolf.position[j] - population[i].position[j])
                Xnew[j] = ac*X1[j] + bc*X2[j] + gc*X3[j]

            #for j in range(dim):
            #   Xnew[j] /= 3.0

            # fitness calculation of new solution
            fnew = fitness.Y(Xnew, fnum)

            # greedy selection
            if fnew < population[i].fitness:
                population[i].position = Xnew
                population[i].fitness = fnew

        # On the basis of fitness values of wolves
        # sort the population in asc order
        population = sorted(population, key=lambda temp: temp.fitness)

        # best 3 solutions will be called as
        # alpha, beta and gama
        alpha_wolf, beta_wolf, gamma_wolf = copy.copy(population[: 3])

        Iter += 1
    # end-while

    # returning the best solution
    return alpha_wolf.position

# variable weight  grey wolf optimization (VMGWO) Gao 2019 =====>
def vmgwo(fnum, max_iter, n, minx, maxx):
    rnd = random.Random(0)

    # create n random wolves
    population = [wolf(fnum, minx, maxx, i) for i in range(n)]

    # On the basis of fitness values of wolves
    # sort the population in asc order
    population = sorted(population, key=lambda temp: temp.fitness)

    # best 3 solutions will be called as
    # alpha, beta and gama
    alpha_wolf, beta_wolf, gamma_wolf = copy.copy(population[: 3])

    # main loop of gwo
    Iter = 0
    while Iter < max_iter:

        # after every 10 iterations
        # print iteration number and best fitness value so far
        # if Iter % 10 == 0 and Iter > 1:
        # print("Iter = " + str(Iter) + " best fitness = %.3f" % alpha_wolf.fitness)

        # linearly decreased from 2 to 0
        a= 2*math.exp(-Iter / max_iter) # VM GAO GWO
        #a = 2 * (1 - Iter / max_iter) #GWO / VMGWO
        # a = 2 - 2 * math.tan((math.pi * Iter) / (5 * max_iter)) # IGWO
        # a = 2 - math.log10(1 + (6 * 0.8) * (Iter / max_iter))  # LFGWO
        #a = 2 - 2 * ((1 - Iter / max_iter) ** 2)  # NGWO
        #a=2/max_iter*math.sqrt(max_iter**2-(Iter+1)**2) #EGWO

        # updating each population member with the help of best three members
        for i in range(n):
            A1, A2, A3 = a * (2 * rnd.random() - 1), a * (
                    2 * rnd.random() - 1), a * (2 * rnd.random() - 1)
            C1, C2, C3 = 2 * rnd.random(), 2 * rnd.random(), 2 * rnd.random()

            X1 = [0.0 for x in range(dim)]
            X2 = [0.0 for x in range(dim)]
            X3 = [0.0 for x in range(dim)]
            Xnew = [0.0 for x in range(dim)]

            shi=0.5*math.atan(Iter)
            theta= (2/math.pi)*math.acos(1/3)*math.atan(Iter)
            ac = math.cos(theta)
            bc = 0.5*math.sin(theta)*math.cos(shi)
            gc = 1-ac-bc

            for j in range(dim):
                X1[j] = alpha_wolf.position[j] - A1 * abs(
                    C1 * alpha_wolf.position[j] - population[i].position[j])
                X2[j] = beta_wolf.position[j] - A2 * abs(
                    C2 * beta_wolf.position[j] - population[i].position[j])
                X3[j] = gamma_wolf.position[j] - A3 * abs(
                    C3 * gamma_wolf.position[j] - population[i].position[j])
                Xnew[j] = ac*X1[j] + bc*X2[j] + gc*X3[j]

            #for j in range(dim):
            #   Xnew[j] /= 3.0

            # fitness calculation of new solution
            fnew = fitness.Y(Xnew, fnum)

            # greedy selection
            if fnew < population[i].fitness:
                population[i].position = Xnew
                population[i].fitness = fnew

        # On the basis of fitness values of wolves
        # sort the population in asc order
        population = sorted(population, key=lambda temp: temp.fitness)

        # best 3 solutions will be called as
        # alpha, beta and gama
        alpha_wolf, beta_wolf, gamma_wolf = copy.copy(population[: 3])

        Iter += 1
    # end-while

    # returning the best solution
    return alpha_wolf.position

# Adaptive  grey wolf optimization (AGWO) Zhang 2021 =============>>>>
def agwo(fnum, max_iter, n, minx, maxx):
    rnd = random.Random(0)

    # create n random wolves
    population = [wolf(fnum, minx, maxx, i) for i in range(n)]

    # On the basis of fitness values of wolves
    # sort the population in asc order
    population = sorted(population, key=lambda temp: temp.fitness)

    # best 3 solutions will be called as
    # alpha, beta and gama
    alpha_wolf, beta_wolf, gamma_wolf = copy.copy(population[: 3])

    # main loop of gwo
    Iter = 0
    while Iter < max_iter:

        # after every 10 iterations
        # print iteration number and best fitness value so far
        # if Iter % 10 == 0 and Iter > 1:
        # print("Iter = " + str(Iter) + " best fitness = %.3f" % alpha_wolf.fitness)

        # linearly decreased from 2 to 0
        # a = 2 * (1 - Iter / max_iter) #GWO
        # a = 2 - 2 * math.tan((math.pi * Iter) / (5 * max_iter)) # IGWO
        # a = 2 - math.log10(1 + (6 * 0.8) * (Iter / max_iter))  # LFGWO
        #a = 2 - 2 * ((1 - Iter / max_iter) ** 2)  # NGWO
        #a=2/max_iter*math.sqrt(max_iter**2-(Iter+1)**2) #EGWO
        #AGWO
        A=np.zeros(n)
        B = np.zeros(n)
        C = np.zeros(n)

        for x in range (n):
            var1=0
            var2=0
            var3=0
            for y in range(dim):
                var1 += (population[x].position[y]-alpha_wolf.position[y])**2
                var2 += (population[x].position[y] - beta_wolf.position[y]) ** 2
                var3 += (population[x].position[y] - gamma_wolf.position[y]) ** 2

            A[x] = math.sqrt(var1) / dim
            B[x] = math.sqrt(var2) / dim
            C[x] = math.sqrt(var3) / dim

        dist1max=A.max()
        dist1avg=np.average(A)
        lamda1=(dist1max-dist1avg)/dist1max
        dist1max = B.max()
        dist1avg = np.average(B)
        lamda2 = (dist1max - dist1avg) / dist1max
        dist1max = C.max()
        dist1avg = np.average(C)
        lamda3 = (dist1max - dist1avg) / dist1max

        a=2-np.log10(1+6*lamda1*(Iter/max_iter))
        # updating each population member with the help of best three members
        for i in range(n):
            A1, A2, A3 = a * (2 * rnd.random() - 1), a * (
                    2 * rnd.random() - 1), a * (2 * rnd.random() - 1)
            C1, C2, C3 = 2 * rnd.random(), 2 * rnd.random(), 2 * rnd.random()

            X1 = [0.0 for x in range(dim)]
            X2 = [0.0 for x in range(dim)]
            X3 = [0.0 for x in range(dim)]
            Xnew = [0.0 for x in range(dim)]

            ac = (lamda1)/(lamda1+lamda2+lamda3)
            bc = (lamda2)/(lamda1+lamda2+lamda3)
            gc =  (lamda3)/(lamda1+lamda2+lamda3)
            for j in range(dim):
                X1[j] = alpha_wolf.position[j] - A1 * abs(
                    C1 * alpha_wolf.position[j] - population[i].position[j])
                X2[j] = beta_wolf.position[j] - A2 * abs(
                    C2 * beta_wolf.position[j] - population[i].position[j])
                X3[j] = gamma_wolf.position[j] - A3 * abs(
                    C3 * gamma_wolf.position[j] - population[i].position[j])

                Xnew[j] = (ac*X1[j] + bc*X2[j] + gc*X3[j])/3

            #for j in range(dim):
            #   Xnew[j] /= 3.0

            # fitness calculation of new solution
            fnew = fitness.Y(Xnew, fnum)

            # greedy selection
            if fnew < population[i].fitness:
                population[i].position = Xnew
                population[i].fitness = fnew

        # On the basis of fitness values of wolves
        # sort the population in asc order
        population = sorted(population, key=lambda temp: temp.fitness)

        # best 3 solutions will be called as
        # alpha, beta and gama
        alpha_wolf, beta_wolf, gamma_wolf = copy.copy(population[: 3])

        Iter += 1
    # end-while

    # returning the best solution
    return alpha_wolf.position

# Improved  grey wolf optimization (TLFGWO) 2019 Yan =>>>>>>>>>>>>>>
def tlfgwo(fnum, max_iter, n, minx, maxx):
    rnd = random.Random(0)

    # create n random wolves
    population = [wolf(fnum, minx, maxx, i) for i in range(n)]

    # On the basis of fitness values of wolves
    # sort the population in asc order
    population = sorted(population, key=lambda temp: temp.fitness)

    # best 3 solutions will be called as
    # alpha, beta and gama
    alpha_wolf, beta_wolf, gamma_wolf = copy.copy(population[: 3])

    # main loop of gwo
    Iter = 0
    while Iter < max_iter:

        # after every 10 iterations
        # print iteration number and best fitness value so far
        # if Iter % 10 == 0 and Iter > 1:
        # print("Iter = " + str(Iter) + " best fitness = %.3f" % alpha_wolf.fitness)

        # linearly decreased from 2 to 0
        # a = 2 * (1 - Iter / max_iter) #GWO
        # a = 2 - 2 * math.tan((math.pi * Iter) / (5 * max_iter)) # IGWO
        # a = 2 - math.log10(1 + (6 * 0.8) * (Iter / max_iter))  # LFGWO
        #a = 2 - 2 * ((1 - Iter / max_iter) ** 2)  # NGWO
        #a=2/max_iter*math.sqrt(max_iter**2-(Iter+1)**2) #EGWO
        #a=2*math.exp(-Iter/max_iter) #VWGWO
        a=2- (math.log10(1+1.3*math.tan(Iter/max_iter)**3))**6 #IGWO1

        # updating each population member with the help of best three members
        for i in range(n):
            A1, A2, A3 = a * (2 * rnd.random() - 1), a * (
                    2 * rnd.random() - 1), a * (2 * rnd.random() - 1)
            C1, C2, C3 = 2 * rnd.random(), 2 * rnd.random(), 2 * rnd.random()

            X1 = [0.0 for x in range(dim)]
            X2 = [0.0 for x in range(dim)]
            X3 = [0.0 for x in range(dim)]
            Xnew = [0.0 for x in range(dim)]

            ac = (alpha_wolf.fitness)/(beta_wolf.fitness+gamma_wolf.fitness)
            bc = (beta_wolf.fitness)/(alpha_wolf.fitness+gamma_wolf.fitness)
            gc = (gamma_wolf.fitness)/(beta_wolf.fitness+alpha_wolf.fitness)
            for j in range(dim):
                X1[j] = alpha_wolf.position[j] - A1 * abs(
                    C1 * alpha_wolf.position[j] - population[i].position[j])
                X2[j] = beta_wolf.position[j] - A2 * abs(
                    C2 * beta_wolf.position[j] - population[i].position[j])
                X3[j] = gamma_wolf.position[j] - A3 * abs(
                    C3 * gamma_wolf.position[j] - population[i].position[j])
                Xnew[j] = (ac*X1[j] + bc*X2[j] + gc*X3[j])/3

            #for j in range(dim):
            #   Xnew[j] /= 3.0

            # fitness calculation of new solution
            fnew = fitness.Y(Xnew, fnum)

            # greedy selection
            if fnew < population[i].fitness:
                population[i].position = Xnew
                population[i].fitness = fnew

        # On the basis of fitness values of wolves
        # sort the population in asc order
        population = sorted(population, key=lambda temp: temp.fitness)

        # best 3 solutions will be called as
        # alpha, beta and gama
        alpha_wolf, beta_wolf, gamma_wolf = copy.copy(population[: 3])

        Iter += 1
    # end-while

    # returning the best solution
    return alpha_wolf.position

# Improved  grey wolf optimization 2 (EGWO) 2020 Galeb ============>>>>>>>>
def igwodsr(fnum, max_iter, n, minx, maxx): # 2020 Galeb ============>>>>>>>>
    rnd = random.Random(0)

    # create n random wolves
    population = [wolf(fnum, minx, maxx, i) for i in range(n)]

    # On the basis of fitness values of wolves
    # sort the population in asc order
    population = sorted(population, key=lambda temp: temp.fitness)

    # best 3 solutions will be called as
    # alpha, beta and gama
    alpha_wolf, beta_wolf, gamma_wolf = copy.copy(population[: 3])

    # main loop of gwo
    Iter = 0
    while Iter < max_iter:

        # after every 10 iterations
        # print iteration number and best fitness value so far
        # if Iter % 10 == 0 and Iter > 1:
        # print("Iter = " + str(Iter) + " best fitness = %.3f" % alpha_wolf.fitness)

        # linearly decreased from 2 to 0
        # a = 2 * (1 - Iter / max_iter) #GWO
        # a = 2 - 2 * math.tan((math.pi * Iter) / (5 * max_iter)) # IGWO
        # a = 2 - math.log10(1 + (6 * 0.8) * (Iter / max_iter))  # LFGWO
        #a = 2 - 2 * ((1 - Iter / max_iter) ** 2)  # NGWO
        #a=2/max_iter*math.sqrt(max_iter**2-(Iter+1)**2) #EGWO
        #a=2*math.exp(-Iter/max_iter) #VWGWO
        #a=2- (math.log10(1+1.3*math.tan(Iter/max_iter)**3))**6 #IGWO1
        a=2*math.exp(-2*(1-(Iter/max_iter))) #IGWO2

        # updating each population member with the help of best three members
        for i in range(n):
            A1, A2, A3 = a * (2 * rnd.random() - 1), a * (
                    2 * rnd.random() - 1), a * (2 * rnd.random() - 1)
            C1, C2, C3 = 2 * rnd.random(), 2 * rnd.random(), 2 * rnd.random()

            X1 = [0.0 for x in range(dim)]
            X2 = [0.0 for x in range(dim)]
            X3 = [0.0 for x in range(dim)]
            Xnew = [0.0 for x in range(dim)]

            F=alpha_wolf.fitness + beta_wolf.fitness+ gamma_wolf.fitness
            if F==0.0:
                F=1
            ac = (alpha_wolf.fitness)/F
            bc = (beta_wolf.fitness)/F
            gc = (gamma_wolf.fitness)/F
            for j in range(dim):
                X1[j] = alpha_wolf.position[j] - A1 * abs(
                    C1 * alpha_wolf.position[j] - population[i].position[j])
                X2[j] = beta_wolf.position[j] - A2 * abs(
                    C2 * beta_wolf.position[j] - population[i].position[j])
                X3[j] = gamma_wolf.position[j] - A3 * abs(
                    C3 * gamma_wolf.position[j] - population[i].position[j])
                #print('ac',ac,'bc', bc, 'gc',gc, 'F', F)
                Xnew[j] = (ac*X1[j] + bc*X2[j] + gc*X3[j])/3


            # fitness calculation of new solution
            #print('IGWODSR, Xnew',Xnew )
            fnew = fitness.Y(Xnew, fnum)

            # greedy selection
            if fnew < population[i].fitness:
                population[i].position = Xnew
                population[i].fitness = fnew

        # On the basis of fitness values of wolves
        # sort the population in asc order
        population = sorted(population, key=lambda temp: temp.fitness)

        # best 3 solutions will be called as
        # alpha, beta and gama
        alpha_wolf, beta_wolf, gamma_wolf = copy.copy(population[: 3])

        Iter += 1
    # end-while

    # returning the best solution
    return alpha_wolf.position

# Modified  grey wolf optimization (MGWO) 2021 kumar   ===========>>>>>>>>>>>
def mgwo(fnum, max_iter, n, minx, maxx): # 2021 Kumar
    rnd = random.Random(0)

    # create n random wolves
    population = [wolf(fnum, minx, maxx, i) for i in range(n)]

    # On the basis of fitness values of wolves
    # sort the population in asc order
    population = sorted(population, key=lambda temp: temp.fitness)

    # best 3 solutions will be called as
    # alpha, beta and gama
    alpha_wolf, beta_wolf, delta_wolf, gamma_wolf = copy.copy(population[: 4])

    # main loop of gwo
    Iter = 0
    while Iter < max_iter:

        # after every 10 iterations
        # print iteration number and best fitness value so far
        # if Iter % 10 == 0 and Iter > 1:
        # print("Iter = " + str(Iter) + " best fitness = %.3f" % alpha_wolf.fitness)

        # linearly decreased from 2 to 0
        # a = 2 * (1 - Iter / max_iter) #GWO
        # a = 2 - 2 * math.tan((math.pi * Iter) / (5 * max_iter)) # IGWO
        # a = 2 - math.log10(1 + (6 * 0.8) * (Iter / max_iter))  # LFGWO
        #a = 2 - 2 * ((1 - Iter / max_iter) ** 2)  # NGWO
        #a=2/max_iter*math.sqrt(max_iter**2-(Iter+1)**2) #EGWO
        #a=2*math.exp(-Iter/max_iter) #VWGWO
        a = 2 * math.exp(-Iter / max_iter)

        # updating each population member with the help of best three members
        for i in range(n):
            A1, A2, A3 = a * (2 * rnd.random() - 1), a * (
                    2 * rnd.random() - 1), a * (2 * rnd.random() - 1)
            C1, C2, C3 = 2 * rnd.random(), 2 * rnd.random(), 2 * rnd.random()

            X1 = [0.0 for x in range(dim)]
            X2 = [0.0 for x in range(dim)]
            X3 = [0.0 for x in range(dim)]
            X4 = [0.0 for x in range(dim)]
            Xnew = [0.0 for x in range(dim)]
                # Need more attention
            theta= (2/math.pi)*math.acos(1/3)*math.atan(Iter)
            ac = math.cos(theta)
            bc = 0.5*math.sin(theta)*math.cos(90-theta)
            gc = 1 - ac - bc

            for j in range(dim):
                X1[j] = alpha_wolf.position[j] - A1 * abs(
                    C1 * alpha_wolf.position[j] - population[i].position[j])
                X2[j] = beta_wolf.position[j] - A2 * abs(
                    C2 * beta_wolf.position[j] - population[i].position[j])
                X3[j] = delta_wolf.position[j] - A3 * abs(
                    C3 * delta_wolf.position[j] - population[i].position[j])
                X4[j] = gamma_wolf.position[j] - A3 * abs(
                    C3 * gamma_wolf.position[j] - population[i].position[j])
                X3[j]=(X3[j]+X4[j])/2
                Xnew[j] = ac*X1[j] + bc*X2[j] + gc*X3[j]

            #for j in range(dim):
            #   Xnew[j] /= 3.0

            # fitness calculation of new solution
            fnew = fitness.Y(Xnew, fnum)

            # greedy selection
            if fnew < population[i].fitness:
                population[i].position = Xnew
                population[i].fitness = fnew

        # On the basis of fitness values of wolves
        # sort the population in asc order
        population = sorted(population, key=lambda temp: temp.fitness)

        # best 3 solutions will be called as
        # alpha, beta and gama
        alpha_wolf, beta_wolf, delta_wolf, gamma_wolf = copy.copy(population[: 4])

        Iter += 1
    # end-while

    # returning the best solution
    return alpha_wolf.position

# enhance exploration  grey wolf optimization (EEGWO) 2018 Long ========>
def eegwo(fnum, max_iter, n, minx, maxx):
    rnd = random.Random(0)

    # create n random wolves
    population = [wolf(fnum, minx, maxx, i) for i in range(n)]

    # On the basis of fitness values of wolves
    # sort the population in asc order
    population = sorted(population, key=lambda temp: temp.fitness)

    # best 3 solutions will be called as
    # alpha, beta and gama
    alpha_wolf, beta_wolf, gamma_wolf = copy.copy(population[: 3])

    # main loop of gwo
    Iter = 0
    while Iter < max_iter:

        # after every 10 iterations
        # print iteration number and best fitness value so far
        # if Iter % 10 == 0 and Iter > 1:
        # print("Iter = " + str(Iter) + " best fitness = %.3f" % alpha_wolf.fitness)

        # linearly decreased from 2 to 0
        # a = 2 * (1 - Iter / max_iter) #GWO
        # a = 2 - 2 * math.tan((math.pi * Iter) / (5 * max_iter)) # IGWO
        # a = 2 - math.log10(1 + (6 * 0.8) * (Iter / max_iter))  # LFGWO
        #a = 2 - 2 * ((1 - Iter / max_iter) ** 2)  # NGWO
        #a=2/max_iter*math.sqrt(max_iter**2-(Iter+1)**2) #EGWO

        a=2-2*((max_iter-Iter)/max_iter)**1.5 #mu=1.5

        # updating each population member with the help of best three members
        for i in range(n):
            A1, A2, A3 = a * (2 * rnd.random() - 1), a * (
                    2 * rnd.random() - 1), a * (2 * rnd.random() - 1)
            C1, C2, C3 = 2 * rnd.random(), 2 * rnd.random(), 2 * rnd.random()

            X1 = [0.0 for x in range(dim)]
            X2 = [0.0 for x in range(dim)]
            X3 = [0.0 for x in range(dim)]
            X4 = [0.0 for x in range(dim)]
            Xnew = [0.0 for x in range(dim)]

            #ac = 1-0.5*(Iter/max_iter)
            #bc =0+ 0.3*(Iter/max_iter)
            #gc = 1-ac-bc
            #EEGWO Position update part
            r = list(range(0,i)) + list(range(i+1,n))
            ch=random.choice(r)
            r3=rnd.random()
            r4=rnd.random()
            b1=1
            b2=0.9
            for j in range(dim):
                X1[j] = alpha_wolf.position[j] - A1 * abs(
                    C1 * alpha_wolf.position[j] - population[i].position[j])
                X2[j] = beta_wolf.position[j] - A2 * abs(
                    C2 * beta_wolf.position[j] - population[i].position[j])
                X3[j] = gamma_wolf.position[j] - A3 * abs(
                    C3 * gamma_wolf.position[j] - population[i].position[j])
                X4[j] = population[ch].position[j]- population[i].position[j]
                Xnew[j] = b1*r3*(X1[j]+X2[j]+X3[j])/3 + b2*r4*X4[j]

                #Xnew[j] += ac*X1[j] + bc*X2[j] + gc*X3[j]

            #for j in range(dim):
            #   Xnew[j] /= 3.0

            # fitness calculation of new solution
            fnew = fitness.Y(Xnew, fnum)

            # greedy selection
            if fnew < population[i].fitness:
                population[i].position = Xnew
                population[i].fitness = fnew

        # On the basis of fitness values of wolves
        # sort the population in asc order
        population = sorted(population, key=lambda temp: temp.fitness)

        # best 3 solutions will be called as
        # alpha, beta and gama
        alpha_wolf, beta_wolf, gamma_wolf = copy.copy(population[: 3])

        Iter += 1
    # end-while

    # returning the best solution
    return alpha_wolf.position
# daptive exploration  grey wolf optimization (AgGWO) 2018 Qais  ======>
def aggwo(fnum, max_iter, n, minx, maxx):
    rnd = random.Random(0)

    # create n random wolves
    population = [wolf(fnum, minx, maxx, i) for i in range(n)]

    # On the basis of fitness values of wolves
    # sort the population in asc order
    population = sorted(population, key=lambda temp: temp.fitness)

    # best 3 solutions will be called as
    # alpha, beta and gama
    alpha_wolf, beta_wolf, gamma_wolf = copy.copy(population[: 3])

    # main loop of gwo
    Iter = 0
    while Iter < max_iter:

        # after every 10 iterations
        # print iteration number and best fitness value so far
        # if Iter % 10 == 0 and Iter > 1:
        # print("Iter = " + str(Iter) + " best fitness = %.3f" % alpha_wolf.fitness)

        # linearly decreased from 2 to 0
        # a = 2 * (1 - Iter / max_iter) #GWO
        # a = 2 - 2 * math.tan((math.pi * Iter) / (5 * max_iter)) # IGWO
        # a = 2 - math.log10(1 + (6 * 0.8) * (Iter / max_iter))  # LFGWO
        #a = 2 - 2 * ((1 - Iter / max_iter) ** 2)  # NGWO
        #a=2/max_iter*math.sqrt(max_iter**2-(Iter+1)**2) #EGWO
        a=2-math.cos(np.random.rand())*(Iter/max_iter)

        # updating each population member with the help of best three members
        for i in range(n):
            A1, A2, A3 = a * (2 * rnd.random() - 1), a * (
                    2 * rnd.random() - 1), a * (2 * rnd.random() - 1)
            C1, C2, C3 = 2 * rnd.random(), 2 * rnd.random(), 2 * rnd.random()

            X1 = [0.0 for x in range(dim)]
            X2 = [0.0 for x in range(dim)]
            X3 = [0.0 for x in range(dim)]
            Xnew = [0.0 for x in range(dim)]


            for j in range(dim):
                X1[j] = alpha_wolf.position[j] - A1 * abs(
                    C1 * alpha_wolf.position[j] - population[i].position[j])
                X2[j] = beta_wolf.position[j] - A2 * abs(
                    C2 * beta_wolf.position[j] - population[i].position[j])
                X3[j] = gamma_wolf.position[j] - A3 * abs(
                    C3 * gamma_wolf.position[j] - population[i].position[j])
                Xnew[j] = (X1[j] + X2[j] )/2

            #for j in range(dim):
            #   Xnew[j] /= 3.0

            # fitness calculation of new solution
            fnew = fitness.Y(Xnew, fnum)

            # greedy selection
            if fnew < population[i].fitness:
                population[i].position = Xnew
                population[i].fitness = fnew

        # On the basis of fitness values of wolves
        # sort the population in asc order
        population = sorted(population, key=lambda temp: temp.fitness)

        # best 3 solutions will be called as
        # alpha, beta and gama
        alpha_wolf, beta_wolf, gamma_wolf = copy.copy(population[: 3])

        Iter += 1
    # end-while

    # returning the best solution
    return alpha_wolf.position

# Beetle Antenna grey wolf optimization (GWO) 2021 Fan =============>
def bgwo(fnum, max_iter, n, minx, maxx):
    rnd = random.Random(0)

    # create n random wolves
    population = [wolf(fnum, minx, maxx, i) for i in range(n)]

    # On the basis of fitness values of wolves
    # sort the population in asc order
    population = sorted(population, key=lambda temp: temp.fitness)

    # best 3 solutions will be called as
    # alpha, beta and gama
    alpha_wolf, beta_wolf, gamma_wolf = copy.copy(population[: 3])

    # main loop of gwo
    Iter = 0
    # DIR
    R=2
    nita=0.95
    SL=2*(maxx-minx)

    while Iter < max_iter:

        # after every 10 iterations
        # print iteration number and best fitness value so far
        #if Iter % 10 == 0 and Iter > 1:
            #print("Iter = " + str(Iter) + " best fitness = %.3f" % alpha_wolf.fitness)

        #DIR ================>
        randlist = np.zeros(dim)
        for i in range(dim):
            randlist[i] = random.uniform(-1, 1)
        dir = copy.deepcopy(randlist / np.linalg.norm(randlist))
        DL = SL / R
        XAL = [0.0 for x in range(dim)]
        XAR = [0.0 for x in range(dim)]
        XNW = [0.0 for x in range(dim)]
        for j in range(dim):
            XAL[j] = alpha_wolf.position[j] -  DL * dir[j]
            XAR[j] = alpha_wolf.position[j] +  DL * dir[j]
        sign=np.sign(fitness.Y(XAL, fnum)-fitness.Y(XAR, fnum))
        for j in range(dim):
            XNW[j] = alpha_wolf.position[j] + SL*dir[j]*sign

        if fitness.Y(XNW, fnum)<=fitness.Y(alpha_wolf.position, fnum):
            alpha_wolf.position=XNW

        SL=nita*SL+0.01
        # linearly decreased from 2 to 0
        a = 2*math.cos((math.pi/2)*(Iter/max_iter)**4)

        # updating each population member with the help of best three members
        for i in range(n):
            A1, A2, A3 = a * (2 * rnd.random() - 1), a * (
                    2 * rnd.random() - 1), a * (2 * rnd.random() - 1)
            C1, C2, C3 = 2 * rnd.random(), 2 * rnd.random(), 2 * rnd.random()

            X1 = [0.0 for x in range(dim)]
            X2 = [0.0 for x in range(dim)]
            X3 = [0.0 for x in range(dim)]
            Xnew = [0.0 for x in range(dim)]
            for j in range(dim):
                X1[j] = alpha_wolf.position[j] - A1 * abs(
                    C1 * alpha_wolf.position[j] - population[i].position[j])
                X2[j] = beta_wolf.position[j] - A2 * abs(
                    C2 * beta_wolf.position[j] - population[i].position[j])
                X3[j] = gamma_wolf.position[j] - A3 * abs(
                    C3 *gamma_wolf.position[j] - population[i].position[j])
                Xnew[j] = X1[j] + X2[j] + X3[j]

            for j in range(dim):
                Xnew[j] /= 3.0

            # fitness calculation of new solution
            fnew = fitness.Y(Xnew, fnum)

            # greedy selection
            if fnew < population[i].fitness:
                population[i].position = Xnew
                population[i].fitness = fnew

        # On the basis of fitness values of wolves
        # sort the population in asc order
        population = sorted(population, key=lambda temp: temp.fitness)

        # best 3 solutions will be called as
        # alpha, beta and gama
        alpha_wolf, beta_wolf, gamma_wolf = copy.copy(population[: 3])

        Iter += 1
    # end-while

    # returning the best solution
    return alpha_wolf.position

# improved by lion  grey wolf optimization (EGWO) Liu 2021 ===============>
def ilgwo(fnum, max_iter, n, minx, maxx):
    rnd = random.Random(0)

    # create n random wolves
    population = [wolf(fnum, minx, maxx, i) for i in range(n)]

    # On the basis of fitness values of wolves
    # sort the population in asc order
    population = sorted(population, key=lambda temp: temp.fitness)

    # best 3 solutions will be called as
    # alpha, beta and gama
    alpha_wolf, beta_wolf, gamma_wolf = copy.copy(population[: 3])

    # main loop of gwo
    Iter = 0
    while Iter < max_iter:

        # after every 10 iterations
        # print iteration number and best fitness value so far
        # if Iter % 10 == 0 and Iter > 1:
        # print("Iter = " + str(Iter) + " best fitness = %.3f" % alpha_wolf.fitness)

        # linearly decreased from 2 to 0
        a = 2 * (1 - Iter / max_iter) #GWO
        # a = 2 - 2 * math.tan((math.pi * Iter) / (5 * max_iter)) # IGWO
        # a = 2 - math.log10(1 + (6 * 0.8) * (Iter / max_iter))  # LFGWO
        #a = 2 - 2 * ((1 - Iter / max_iter) ** 2)  # NGWO
        #a=2/max_iter*math.sqrt(max_iter**2-(Iter+1)**2) #EGWO

        # updating each population member with the help of best three members
        alpha_f= math.exp((-30*Iter)/max_iter)**10
        alpha_c=(max_iter-Iter)/max_iter
        beta_1= rnd.random()

        for i in range(n):
            A1, A2, A3 = a * (2 * rnd.random() - 1), a * (
                    2 * rnd.random() - 1), a * (2 * rnd.random() - 1)
            C1, C2, C3 = 2 * rnd.random(), 2 * rnd.random(), 2 * rnd.random()

            X1 = [0.0 for x in range(dim)]
            X2 = [0.0 for x in range(dim)]
            X3 = [0.0 for x in range(dim)]
            Xnew = [0.0 for x in range(dim)]


            for j in range(dim):
                if a<0.5:
                    X1[j] = alpha_wolf.position[j] - A1 * abs(C1 * alpha_wolf.position[j] - population[i].position[j])
                    X2[j] = beta_wolf.position[j] - A2 * abs(C2 * beta_wolf.position[j] - population[i].position[j])
                    X3[j] = gamma_wolf.position[j] - A3 * abs(C3 * gamma_wolf.position[j] - population[i].position[j])
                    Xnew[j] = (X1[j] + X2[j] + X3[j])/3
                else:
                    X1[j] = (alpha_wolf.position[j] - A1 * abs(C1 * alpha_wolf.position[j] - population[i].position[j]))*alpha_f
                    X2[j] = (beta_wolf.position[j] - A2 * abs(C2 * beta_wolf.position[j] - population[i].position[j]))*beta_1
                    X3[j] = (gamma_wolf.position[j] - A3 * abs(C3 * gamma_wolf.position[j] - population[i].position[j]))*alpha_c
                    Xnew[j] = (X1[j] + X2[j] + X3[j]) / (3+C1)

            #for j in range(dim):
            #   Xnew[j] /= 3.0

            # fitness calculation of new solution
            fnew = fitness.Y(Xnew, fnum)

            # greedy selection
            if fnew < population[i].fitness:
                population[i].position = Xnew
                population[i].fitness = fnew

        # On the basis of fitness values of wolves
        # sort the population in asc order
        population = sorted(population, key=lambda temp: temp.fitness)

        # best 3 solutions will be called as
        # alpha, beta and gama
        alpha_wolf, beta_wolf, gamma_wolf = copy.copy(population[: 3])

        Iter += 1
    # end-while

    # returning the best solution
    return alpha_wolf.position

def rwgwo(fnum, max_iter, n, minx, maxx):


    rnd = random.Random(0)

    # create n random wolves
    population = [wolf(fnum, minx, maxx, i) for i in range(n)]

    # On the basis of fitness values of wolves
    # sort the population in asc order
    population = sorted(population, key=lambda temp: temp.fitness)

    # best 3 solutions will be called as
    # alpha, beta and gama
    alpha_wolf, beta_wolf, gamma_wolf = copy.copy(population[: 3])

    # main loop of gwo
    Iter = 0
    while Iter < max_iter:

        # after every 10 iterations
        # print iteration number and best fitness value so far
        #if Iter % 10 == 0 and Iter > 1:
            #print("Iter = " + str(Iter) + " best fitness = %.3f" % alpha_wolf.fitness)

        # linearly decreased from 2 to 0
        a = 2 * (1 - Iter / max_iter)

        A_RND = [0.0 for x in range(dim)]
        B_RND = [0.0 for x in range(dim)]
        G_RND = [0.0 for x in range(dim)]
        V = a * np.random.standard_cauchy()
        for j in range(dim):
            A_RND[j] = alpha_wolf.position[j] + V
            B_RND[j] = beta_wolf.position[j] + V
            G_RND[j] = gamma_wolf.position[j] + V
        if (fitness.Y(A_RND, fnum)) < alpha_wolf.fitness:
            alpha_wolf.position = A_RND
            alpha_wolf.fitness = fitness.Y(A_RND, fnum)
        if (fitness.Y(B_RND, fnum)) < beta_wolf.fitness:
            beta_wolf.position = B_RND
            beta_wolf.fitness = fitness.Y(B_RND, fnum)
        if (fitness.Y(G_RND, fnum)) < gamma_wolf.fitness:
            gamma_wolf.position = G_RND
            gamma_wolf.fitness = fitness.Y(G_RND, fnum)


        # updating each population member with the help of best three members
        for i in range(n):
            A1, A2, A3 = a * (2 * rnd.random() - 1), a * (
                    2 * rnd.random() - 1), a * (2 * rnd.random() - 1)
            C1, C2, C3 = 2 * rnd.random(), 2 * rnd.random(), 2 * rnd.random()

            X1 = [0.0 for x in range(dim)]
            X2 = [0.0 for x in range(dim)]
            X3 = [0.0 for x in range(dim)]
            Xnew = [0.0 for x in range(dim)]
            for j in range(dim):
                X1[j] = alpha_wolf.position[j] - A1 * abs(
                    C1 * alpha_wolf.position[j] - population[i].position[j])
                X2[j] = beta_wolf.position[j] - A2 * abs(
                    C2 * beta_wolf.position[j] - population[i].position[j])
                X3[j] = gamma_wolf.position[j] - A3 * abs(
                    C3 * gamma_wolf.position[j] - population[i].position[j])
                Xnew[j] = X1[j] + X2[j] + X3[j]

            for j in range(dim):
                Xnew[j] /= 3.0

            # fitness calculation of new solution
            fnew = fitness.Y(Xnew,fnum)

            # greedy selection
            if fnew < population[i].fitness:
                population[i].position = Xnew
                population[i].fitness = fnew

        # On the basis of fitness values of wolves
        # sort the population in asc order
        population = sorted(population, key=lambda temp: temp.fitness)

        # best 3 solutions will be called as
        # alpha, beta and gama
        alpha_wolf, beta_wolf, gamma_wolf = copy.copy(population[: 3])

        Iter += 1
    # end-while

    # returning the best solution
    return alpha_wolf.position


# elliptical  grey wolf optimization (EGWO)
def egwo(fnum, max_iter, n, minx, maxx):
    rnd = random.Random(0)

    # create n random wolves
    population = [wolf(fnum, minx, maxx, i) for i in range(n)]

    # On the basis of fitness values of wolves
    # sort the population in asc order
    population = sorted(population, key=lambda temp: temp.fitness)

    # best 3 solutions will be called as
    # alpha, beta and gama
    alpha_wolf, beta_wolf, gamma_wolf, omega1, omega2 = copy.copy(population[: 5])

    # main loop of gwo
    Iter = 0

    while Iter < max_iter:
        a = 2 / max_iter * math.sqrt(max_iter ** 2 - Iter ** 2)

        A_RND = [0.0 for x in range(dim)]
        B_RND = [0.0 for x in range(dim)]
        G_RND = [0.0 for x in range(dim)]

        #V1 = rnd.random() * (-1) ** math.floor(10 * rnd.random())
        #V2 = rnd.random() * (-1) ** math.floor(10 * rnd.random())
        #V3 = rnd.gauss(2, 50) / 100
        for j in range(dim):
            V1 = rnd.random() * (-1) ** math.floor(10 * rnd.random())
            V2 = rnd.random() * (-1) ** math.floor(10 * rnd.random())
            V3 = rnd.random() * (-1) ** math.floor(10 * rnd.random())
            A_RND[j] = alpha_wolf.position[j] + V1
            B_RND[j] = beta_wolf.position[j] + V2
            G_RND[j] = gamma_wolf.position[j] + V3

        fit_A = fitness.Y(A_RND, fnum)
        fit_B = fitness.Y(B_RND, fnum)
        fit_G = fitness.Y(G_RND, fnum)
        if fit_A < alpha_wolf.fitness:
            alpha_wolf.position = A_RND
            alpha_wolf.fitness = fit_A
        if fit_B < beta_wolf.fitness:
            beta_wolf.position = B_RND
            beta_wolf.fitness = fit_B
        if fit_G < gamma_wolf.fitness:
            gamma_wolf.position = G_RND
            gamma_wolf.fitness = fit_G

        # updating each population member with the help of best three members
        for i in range(n):
            A1, A2, A3, = a * (2 * rnd.random() - 1), a * (2 * rnd.random() - 1), a * (2 * rnd.random() - 1)

            #C1 = 2.0+(0.2-2.0)*Iter/max_iter
            C1 = 2 * rnd.random()
            C2 = 2 * rnd.random()
            #C2 = 2.0+(0.2-2.0)*Iter/max_iter
            #C2 = 0.2+(2.0-0.5)*Iter/max_iter
            #C3 = 2.0+(0.2-2.0)*Iter/max_iter
            C3 = 2 * rnd.random()

            X1 = [0.0 for x in range(dim)]
            X2 = [0.0 for x in range(dim)]
            X3 = [0.0 for x in range(dim)]
            Xnew = [0.0 for x in range(dim)]

            gc = math.exp((-Iter)/max_iter)**3
            ac = (1 / (max_iter ) * math.sqrt((max_iter) ** 2 - (Iter + 1) ** 2))
            bc = rnd.random()

            Div=(2 - Iter / max_iter)

            for j in range(dim):
                X1[j] = alpha_wolf.position[j] - A1 * abs(C1 * alpha_wolf.position[j] - population[i].position[j])
                X2[j] = beta_wolf.position[j] - A2 * abs(C2 * beta_wolf.position[j] - population[i].position[j])
                X3[j] = gamma_wolf.position[j] - A3 * abs(C3 * gamma_wolf.position[j] - population[i].position[j])

                if a>=1.0:
                    Xnew[j] = ((ac * X1[j] + bc * X2[j] + gc * X3[j]) / Div)
                else:
                    Xnew[j] = (X1[j]+X2[j])/1.5


            # fitness calculation of new solution
            fnew = fitness.Y(Xnew, fnum)

            # greedy selection
            if fnew < population[i].fitness:
                population[i].position = Xnew
                population[i].fitness = fnew

        # On the basis of fitness values of wolves
        # sort the population in asc order
        population = sorted(population, key=lambda temp: temp.fitness)

        # best 3 solutions will be called as
        # alpha, beta and gama
        alpha_wolf, beta_wolf, gamma_wolf, omega1, omega2 = copy.copy(population[: 5])

        Iter += 1
    # end-while

    # returning the best solution
    return alpha_wolf.position


num_particles = 50
max_iter = 500
LB=-100
UB=100
SimIter=1
for j in range (5):

    gwo_fit = np.zeros(SimIter)
    eegwo_fit = np.zeros(SimIter)
    igwo_fit = np.zeros(SimIter)
    ngwo_fit = np.zeros(SimIter)
    lfgwo_fit = np.zeros(SimIter)
    aggwo_fit = np.zeros(SimIter)
    vmgwo_fit = np.zeros(SimIter)
    tlfgwo_fit = np.zeros(SimIter)
    igwodsr_fit = np.zeros(SimIter)
    bgwo_fit = np.zeros(SimIter)
    agwo_fit = np.zeros(SimIter)
    vwgwo_fit = np.zeros(SimIter)
    mgwo_fit = np.zeros(SimIter)
    ilgwo_fit = np.zeros(SimIter)
    rwgwo_fit = np.zeros(SimIter)
    egwo_fit = np.zeros(SimIter)

    CEC13=pd.DataFrame(data={'Algorithm':['GWO', 'EEGWO', 'IGWO', 'NGWO',
                                       'LFGWO', 'AgGWO', 'VMGWO', 'TLFGWO','IGWO_DSR', 'AGWO',
                                       'VWGWO', 'MGWO', 'ILGWO','RWGWO', 'EGWO']})


    for i in range (1,29):
        fnum = i
        print("Function ", i)
        for j in range (SimIter):
            gwo_fit[j]=fitness.Y(gwo(fnum, max_iter, num_particles, LB, UB),fnum)
            #print(gwo_fit[j])
            eegwo_fit[j] = fitness.Y(eegwo(fnum, max_iter, num_particles, LB, UB),fnum)
            #print(eegwo_fit[j])
            igwo_fit[j] = fitness.Y(igwo(fnum, max_iter, num_particles, LB, UB),fnum)
            ngwo_fit[j] = fitness.Y(ngwo(fnum, max_iter, num_particles, LB, UB),fnum)
            lfgwo_fit[j] = fitness.Y(lfgwo(fnum, max_iter, num_particles, LB, UB),fnum)
            aggwo_fit[j] = fitness.Y(aggwo(fnum, max_iter, num_particles, LB, UB),fnum)
            vmgwo_fit[j] = fitness.Y(vmgwo(fnum, max_iter, num_particles, LB, UB),fnum)
            tlfgwo_fit[j] = fitness.Y(tlfgwo(fnum, max_iter, num_particles, LB, UB),fnum)
            igwodsr_fit[j] = fitness.Y(igwodsr(fnum, max_iter, num_particles, LB, UB),fnum)
            #bgwo_fit[j] = fitness.Y(bgwo(fnum, max_iter, num_particles, LB, UB),fnum)
            #print(bgwo_fit[j])
            agwo_fit[j] = fitness.Y(agwo(fnum, max_iter, num_particles, LB, UB),fnum)
            vwgwo_fit[j] = fitness.Y(vwgwo(fnum, max_iter, num_particles, LB, UB),fnum)
            mgwo_fit[j] = fitness.Y(mgwo(fnum, max_iter, num_particles, LB, UB),fnum)
            ilgwo_fit[j] = fitness.Y(ilgwo(fnum, max_iter, num_particles, LB, UB),fnum)
            rwgwo_fit[j] = fitness.Y(rwgwo(fnum, max_iter, num_particles, LB, UB), fnum)
            egwo_fit[j] = fitness.Y(egwo(fnum, max_iter, num_particles, LB, UB),fnum)
            #print(egwo_fit[j])
        Sim_rlt=[np.average(gwo_fit),np.average(eegwo_fit),
                np.average(igwo_fit),np.average(ngwo_fit),
                np.average(lfgwo_fit),np.average(aggwo_fit),
                np.average(vmgwo_fit), np.average(tlfgwo_fit),
                np.average(igwodsr_fit), np.average(agwo_fit),np.average(vwgwo_fit),
                np.average(mgwo_fit), np.average(ilgwo_fit),
                 np.average(rwgwo_fit),np.average(egwo_fit)]

        Sim_worst = [np.max(gwo_fit),np.max(eegwo_fit),
                    np.max(igwo_fit),np.max(ngwo_fit),
                    np.max(lfgwo_fit),np.max(aggwo_fit),
                    np.max(vmgwo_fit), np.max(tlfgwo_fit),
                    np.max(igwodsr_fit), np.max(agwo_fit),
                    np.max(vwgwo_fit),
                    np.max(mgwo_fit), np.max(ilgwo_fit),
                    np.max(rwgwo_fit),np.max(egwo_fit)]
        Sim_best = [np.min(gwo_fit),np.min(eegwo_fit),
                np.min(igwo_fit),np.min(ngwo_fit),
                np.min(lfgwo_fit),np.min(aggwo_fit),
                np.min(vmgwo_fit), np.min(tlfgwo_fit),
                np.min(igwodsr_fit), np.min(agwo_fit),np.min(vwgwo_fit),
                np.min(mgwo_fit), np.min(ilgwo_fit),
                 np.min(rwgwo_fit),np.min(egwo_fit)]
        Sim_std = [np.std(gwo_fit),np.std(eegwo_fit),
                np.std(igwo_fit),np.std(ngwo_fit),
                np.std(lfgwo_fit),np.std(aggwo_fit),
                np.std(vmgwo_fit), np.std(tlfgwo_fit),
                np.std(igwodsr_fit), np.std(agwo_fit),np.std(vwgwo_fit),
                np.std(mgwo_fit), np.std(ilgwo_fit),
                 np.std(rwgwo_fit),np.std(egwo_fit)]
        CEC13['F'+str(i)]=Sim_rlt
        CEC13['WS_F' + str(i)] = Sim_worst
        CEC13['BS_F' + str(i)] = Sim_best
        CEC13['STD_F' + str(i)] = Sim_std
        CEC13['Rank F'+str(i)]=CEC13['F'+str(i)].rank(ascending=1, method='min').astype(int)
        #print(CEC13[['Algorithm', 'Rank F'+str(i)]])

    avg=pd.DataFrame(CEC13, columns=['Rank F1','Rank F2','Rank F3','Rank F4','Rank F5','Rank F6',
                                     'Rank F7', 'Rank F8','Rank F9','Rank F10','Rank F11','Rank F12',
                                     'Rank F13','Rank F14','Rank F15','Rank F16','Rank F17',
                                    'Rank F18', 'Rank F19', 'Rank F20', 'Rank F21', 'Rank F22', 'Rank F23',
                                     'Rank F24','Rank F25','Rank F26','Rank F27','Rank F28'])
    CEC13['Avg Rank']=avg.mean(axis=1)
    pd.set_option('display.max_columns', None)
    #print(CEC13[['Algorithm', 'Avg Rank']])
    print('Round' ,j)
    print(max_iter,'\t', CEC13.loc[14].at["Avg Rank"])
    CEC13.to_csv(str(num_particles)+'CEC13.txt',sep='\t')
    num_particles=num_particles+50