### 100개의 data를 preprocessing할 것.

### Objective
# 7,0,0,0,0,211,255,201,31,25,0,0,0, ...

# 가...

# input = [0.01, 0.01, ... 0.99, ... ]
# label = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.99, 0.01, 0.01]

# 로 변해야 한다.


### How?
# 1. 이미지 픽셀값 정규화.
# 2. 레이블 생성.


import numpy as np

data_file = open('mnist_test_10.csv', 'r')      # 데이터를 읽어온 후
all_lines = data_file.readlines()               # 모든 행을 리스트로 저장
data_file.close()

all_values = all_lines[0].split(',')                                    # 첫 행의 내용을 불러온 후

label_list = np.zeros(10) + 0.01                                        # 레이블 생성
label_list[int(all_values[0])] = 0.99
# print(label_list)
input_list = ( np.asfarray(all_values[1:]) / 255 * 0.99) + 0.01         # 실제 값은 [0.01, 1.00] 으로 정규화
# print(input_list)


print(np.argmax(label_list))