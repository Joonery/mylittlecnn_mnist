import numpy as np                  # matrix
import scipy.special                # sigmoid
import time                         # performance test
import pandas as pd                 # matrix export / import
import matplotlib.pyplot as plt     # result visualization


# TODO 1 : negative / non-negative weights의 차이를 알아보기 위해, init mehtod를 수정하여 weight가 (0,1]의 값을 갖도록 초기화해보고, 성능을 비교하기.
# TODO 2 : 매 train마다 가중치 matrix의 시각화 사진을 저장해서, 1000회 학습 동안 가중치가 어떻게 변하는지 확인하는 영상 제작.

class Neural_Network() :

    ### Initialize network.
    def __init__(self, input_nodes, hidden_nodes, output_nodes, lr) :

        ### Node 개수
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes

        ### Learning Rate
        self.lr = lr

        ### Matrixes
        # 뒷 layer가 행의 개수가 되어야 함.
        # sigmoid를 사용하므로 초기화는 Xavier method (음의 값도 가지므로, 필요하면 lognormal로 초기화해도 됨.)
                                    # mean, standard variance,      (row, col)
        self.w_i_h = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.w_h_o = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        ### Activation Function
        self.activation_function = lambda x: scipy.special.expit(x)

        ### Reverse Activation Function
        self.reverse_activation_function = lambda x : scipy.special.logit(x)


    ### Train network.
    def train(self, input_lists, label_lists) :
        
        ### preprocessing
        inputs = np.array(input_lists, ndmin=2).T           # 입력받은 리스트를 2차원 행렬로 변환 후 transpose.
        labels = np.array(label_lists, ndmin=2).T           # 입력받은 리스트를 2차원 행렬로 변환 후 transpose.
        

        ### propagation (same with query, but needs by-products)
        hidden_inputs = np.dot(self.w_i_h, inputs)                  # 은닉층 input
        hidden_outputs = self.activation_function(hidden_inputs)    # 은닉층 output

        final_inputs = np.dot(self.w_h_o, hidden_outputs)           # 출력층 input
        final_outputs = self.activation_function(final_inputs)      # 출력층 output


        ### 각 layer별 오차 추출
        output_errors = labels - final_outputs    # output layer에서의 error
        hidden_errors = np.dot(self.w_h_o.T, output_errors)   # hidden layer에서의 error


        ### back-propagation
        self.w_h_o += self.lr * np.dot( (output_errors*final_outputs*(1-final_outputs)), np.transpose(hidden_outputs))
        self.w_i_h += self.lr * np.dot( (hidden_errors*hidden_outputs*(1-hidden_outputs)), np.transpose(inputs))


    ### Query to network. (param : input lists should be list or 1-dimension array!)
    def query(self, input_lists) :

        inputs = np.array(input_lists, ndmin=2).T   # 입력받은 리스트를 2차원 행렬로 변환 후 transpose.

        # 은닉층 input
        hidden_inputs = np.dot(self.w_i_h, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        # 은닉층 output

        # 출력층 input
        final_inputs = np.dot(self.w_h_o, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        # 출력층 output

        return final_outputs


    # 결과 출력기.
    def report(self) :
        pass


    ### 가중치 행렬들을 csv로 내보내기.
    def save_weight_matrix(self) :

        df = pd.DataFrame(self.w_i_h)
        df.to_csv("w_i_h.csv", index=False)

        df = pd.DataFrame(self.w_h_o)
        df.to_csv("w_h_o.csv", index=False)

        print("Weight matrix exported!")


    ### 가중치 행렬들을 csv로 불러오기.
    def load_weight_matrix(self) :

        df = pd.read_csv("w_i_h.csv")
        self.w_i_h = np.array(df)

        df = pd.read_csv("w_h_o.csv")
        self.w_h_o = np.array(df)
        
        print("Weight matrix imported!")


    ### 가중치 행렬을 보여줌.
    def print_weight_matrix(self) :
        print("Input to Hidden matrix :")
        print(self.w_i_h)
        print("Hidden to Output matrix :")
        print(self.w_h_o)


    # 가중치 행렬을 시각회해 보여줌.
    def show_weight_matrix_in_plt(self) :
        pass


    # 역질의
    def reverse_query(self, label_list) :
        
        # 정답지를 세로로 만들어서
        final_outputs = np.array(label_list, ndmin=2).T
        
        # 시그모이드 이전으로 되돌림 (final_input 상태로)
        final_inputs = self.reverse_activation_function(final_outputs)
        
        # output > hidden으로 가기 위해 hidden_output을 계산.
        hidden_outputs = np.dot(self.w_h_o.T, final_inputs)

        # 0.01 - 0.99 의 값으로 정규화
        hidden_outputs -= np.min(hidden_outputs)
        hidden_outputs /= np.max(hidden_outputs)
        hidden_outputs *= 0.98
        hidden_outputs += 0.01
        
        # 시그모이드 이전으로 되돌림 (hidden_input 상태로)
        hidden_inputs = self.reverse_activation_function(hidden_outputs)
        
        # input을 계산
        inputs = np.dot(self.w_i_h.T, hidden_inputs)
        
        # 0.01 - 0.99 의 값으로 정규화
        inputs -= np.min(inputs)
        inputs /= np.max(inputs)
        inputs *= 0.98
        inputs += 0.01
        
        return inputs


# =============================================================================================================================
# =============================================================================================================================
# =============================================================================================================================
# =============================================================================================================================
# =============================================================================================================================


### CONFIG ==================================================================
input_nodes = 784
hidden_nodes = 100
output_nodes = 10
lr = 0.3


### CREATE ==================================================================
my_network = Neural_Network(input_nodes, hidden_nodes, output_nodes, lr)


### TRAIN ==================================================================
data_file = open('mnist_train_100.csv', 'r')
all_lines = data_file.readlines()
data_file.close()

idx = 100
for one_image in all_lines :

    # 해당 행의 값을 리스트로 만들어
    all_values = one_image.split(',')

    # 답지 생성 (답 값만 0.99로.)
    label_list = np.zeros(10) + 0.01
    label_list[int(all_values[0])] = 0.99

    # 문제지 생성 ([0.01, 1.00] 으로 정규화)
    input_list = ( np.asfarray(all_values[1:]) / 255 * 0.99) + 0.01

    # 학습
    my_network.train(input_list, label_list)

    # 보고
    idx -= 1
    print("학습 완료. {}개 남음..".format(idx))
    

### VALIDATON ==================================================================
data_file = open('mnist_test_10.csv', 'r')
all_lines = data_file.readlines()
data_file.close()

answer = []

for one_line in all_lines :

    all_values = one_line.split(',')
    input_list = ( np.asfarray(all_values[1:]) / 255 * 0.99) + 0.01
    result = my_network.query(input_list)

    print("정답 : {}, 예측 : {}".format(all_values[0], np.argmax(result)))
    
    if int(all_values[0]) == int(np.argmax(result)) :
        answer.append(1)
    else :
        answer.append(0)

print("Performance : {} / {} = {} %".format(sum(answer), len(answer), sum(answer)/len(answer)*100))

### EXPORT ==================================================================
# my_network.save_weight_matrix()


### 역질의 ==================================================================

# 0부터 9까지
for i in range(10) :
    
    # 답지를 생성
    label_list = np.zeros(10) + 0.01
    label_list[i] = 0.99
    
    # 역질의
    result = my_network.reverse_query(label_list)

    # 역질의 결과를 그림으로 저장
    image_array = np.asfarray(result.reshape((28,28)))
    plt.imshow(image_array, cmap='Greys', interpolation='None')     # plt에다 데이터를 그린다.
    plt.savefig('./requery_result_{}.png'.format(i))

print('Save completed.')