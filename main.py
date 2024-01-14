import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_cost(x, y, w, b):
    m = x.shape[0]
    if m == 0:
        raise ValueError("Number of training examples is zero.")
    total_cost = 0
    cost_sum = 0
    for i in range(m):
        f_wb = w*x[i] + b
        cost = (f_wb - y[i])**2
        cost_sum += cost

    total_cost = (1/(2*m)) * cost_sum
    return total_cost


def compute_gradient(x, y, w, b):
    m = x.shape[0]

    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = (f_wb - y[i])
        dj_dw += dj_dw_i
        dj_db += dj_db_i
    dj_dw = (1/m) * dj_dw
    dj_db = (1/m) * dj_db

    return dj_dw, dj_db


def gradient_descent(x, y, w_in, b_in, alpha, num_iters):
    m = len(x)
    w = w_in
    b = b_in
    it_arr = []
    cost_arr = []
    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        cost = compute_cost(x, y, w, b)
        it_arr.append(i)
        cost_arr.append(cost)
        if i % 10 == 0:
            print("iteration {} , cost {}".format(i, cost))
    print("Final iteration {} , cost {} , w {} , b {}".format(num_iters, cost, w, b))
    #
    # plt.plot(it_arr, cost_arr)
    # plt.ylabel('Cost')
    # plt.xlabel('Iteration')
    # plt.show()

    return w, b


dataset = pd.read_csv("data/Salary Data.csv")
dataset = dataset[['Years of Experience', 'Salary']]

dataset['Years of Experience'] = pd.to_numeric(dataset['Years of Experience'], errors='coerce')
dataset['Salary'] = pd.to_numeric(dataset['Salary'], errors='coerce')

dataset = dataset.dropna()

x_train = np.array(dataset['Years of Experience'])
y_train = np.array(dataset['Salary'])

print(x_train.shape)
print(y_train.shape)
print("Number of training examples(m):", len(x_train))

# plt.scatter(x_train, y_train, marker='x', c='r')
# plt.title('Salary vs Years of Experience')
# plt.ylabel('Salary')
# plt.xlabel('Years of Experience')
# plt.show()

initial_w = 0
initial_b = 0

cost = compute_cost(x_train, y_train, initial_w, initial_b)
# print(type(cost))
print(f'Cost at initial w: {cost:.3f}')

tmp_dj_dw, tmp_dj_db = compute_gradient(x_train, y_train, initial_w, initial_b)
print("Gradient at initial w, b (zeros):", tmp_dj_dw, tmp_dj_db)

learning_rate = 0.0001
iterations = 1000
op_w, op_b = gradient_descent(x_train, y_train, initial_w, initial_b, learning_rate, iterations)

m = x_train.shape[0]
predicted = np.zeros(m)

for i in range(m):
    predicted[i] = op_w * x_train[i] + op_b

plt.plot(x_train, predicted, c="b")
plt.scatter(x_train, y_train, marker='x', c='r')
plt.title("Years of Experience vs Salary")
plt.ylabel('Salary')
plt.xlabel('Years of Experience')
plt.show()

years = input('Enter years of experience you have:')
years = int(years)
predicted_salary = years * op_w + op_b
rounded_salary = round(predicted_salary,2)
print(f'For {years} years of experience, your predicted salary to be: {rounded_salary}')