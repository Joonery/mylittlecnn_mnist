import numpy as np
import matplotlib.pyplot as plt

data_file = open('mnist_test_10.csv', 'r')      # 데이터를 읽어온 후
all_lines = data_file.readlines()               # 모든 행을 리스트로 저장
data_file.close()

all_values = all_lines[0].split(',')                            # 첫 행의 내용을 불러온 후
image_array = np.asfarray(all_values[1:]).reshape((28,28))      # 28*28 행렬에 float형태로 집어넣은 후
plt.imshow(image_array, cmap='Greys', interpolation='None')     # plt에다 데이터를 그린다.
# plt.show()                                                      # 이를 보여줌.

i = 1
plt.savefig('./fig_{}.png'.format(i))