import numpy as np                  # matrix
import scipy.special                # sigmoid
import pandas as pd                 # matrix export / import
import matplotlib.pyplot as plt     # result visualization


class Neural_Network() :

    ### Initialize network.
    def __init__(self, input_nodes, hidden1_nodes, hidden2_nodes, output_nodes, lr) :

        ### Node 개수
        self.i_nodes = input_nodes
        self.h1_nodes = hidden1_nodes
        self.h2_nodes = hidden2_nodes
        self.o_nodes = output_nodes

        ### Learning Rate
        self.lr = lr

        ### Matrixes
        # sigmoid를 사용하므로 초기화는 Xavier method (음의 값도 가지므로, 필요하면 lognormal로 초기화해도 됨.)
                                    # mean, standard variance,      (row, col)
        self.w_i_h1 = np.random.normal(0.0, pow(self.h1_nodes, -0.5), (self.h1_nodes, self.i_nodes))
        self.w_h1_h2 = np.random.normal(0.0, pow(self.h2_nodes, -0.5), (self.h2_nodes, self.h1_nodes))
        self.w_h2_o = np.random.normal(0.0, pow(self.o_nodes, -0.5), (self.o_nodes, self.h2_nodes))

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
        hlayer1_inputs = np.dot(self.w_i_h1, inputs)                    # 은닉층 1 input
        hlayer1_outputs = self.activation_function(hlayer1_inputs)      # 은닉층 1 output

        hlayer2_inputs = np.dot(self.w_h1_h2, hlayer1_outputs)          # 은닉층 2 input
        hlayer2_outputs = self.activation_function(hlayer2_inputs)      # 은닉층 2 output

        olayer_inputs = np.dot(self.w_h2_o, hlayer2_outputs)            # 출력층 input
        olayer_outputs = self.activation_function(olayer_inputs)        # 출력층 output

        ### 각 layer별 오차 추출
        olayer_errors = labels - olayer_outputs                         # output layer  에서의 error
        hlayer2_errors = np.dot(self.w_h2_o.T, olayer_errors)           # hidden layer 2에서의 error
        hlayer1_errors = np.dot(self.w_h1_h2.T, hlayer2_errors)         # hidden layer 1에서의 error

        ### back-propagation
                                        # (해당 단계의 오차 * 해당 단계의 출력 미분값 )   행렬곱    (전 단계의 출력)
        self.w_h2_o += self.lr * np.dot( (olayer_errors*olayer_outputs*(1-olayer_outputs)), np.transpose(hlayer2_outputs))
        self.w_h1_h2 += self.lr * np.dot( (hlayer2_errors*hlayer2_outputs*(1-hlayer2_outputs)), np.transpose(hlayer1_outputs))
        self.w_i_h1 += self.lr * np.dot( (hlayer1_errors*hlayer1_outputs*(1-hlayer1_outputs)), np.transpose(inputs))


    ### Query to network. (param : input lists should be list or 1-dimension array!)
    def query(self, input_lists) :

        inputs = np.array(input_lists, ndmin=2).T   # 입력받은 리스트를 2차원 행렬로 변환 후 transpose.

        hlayer1_inputs = np.dot(self.w_i_h1, inputs)                    # 은닉층 1 input
        hlayer1_outputs = self.activation_function(hlayer1_inputs)      # 은닉층 1 output

        hlayer2_inputs = np.dot(self.w_h1_h2, hlayer1_outputs)          # 은닉층 2 input
        hlayer2_outputs = self.activation_function(hlayer2_inputs)      # 은닉층 2 output

        olayer_inputs = np.dot(self.w_h2_o, hlayer2_outputs)            # 출력층 input
        olayer_outputs = self.activation_function(olayer_inputs)        # 출력층 output
        
        return olayer_outputs


    # 결과 출력 todo : 무슨 결과?
    def report(self) :
        pass


    ### 가중치 행렬들을 csv로 내보내기.
    def save_weight_matrix(self) :

        df = pd.DataFrame(self.w_i_h1)
        df.to_csv("w_i_h1.csv", index=False)

        df = pd.DataFrame(self.w_h1_h2)
        df.to_csv("w_h1_h2.csv", index=False)

        df = pd.DataFrame(self.w_h2_o)
        df.to_csv("w_h2_o.csv", index=False)

        print("Weight matrixes exported!")


    ### 가중치 행렬들을 csv로 불러오기.
    def load_weight_matrix(self) :

        df = pd.read_csv("w_i_h1.csv")
        self.w_i_h1 = np.array(df)

        df = pd.read_csv("w_h_o.csv")
        self.w_h_o = np.array(df)
        
        print("Weight matrix imported!")


    ### 가중치 행렬을 출력
    def print_weight_matrix(self) :
        print("Input to Hidden1 matrix :")
        print(self.w_i_h1)
        print("Hidden1 to Hidden2 matrix :")
        print(self.w_h1_h2)
        print("Hidden2 to Output matrix :")
        print(self.w_h2_o)


    # 가중치 행렬을 시각화해 보여줌.
    def show_weight_matrix_in_plt(self) :
        pass


    # 역질의
    def reverse_query(self, label_list) :
        
        # 정답지를 세로로 만들어서
        olayer_outputs = np.array(label_list, ndmin=2).T
        # 시그모이드 이전으로 되돌림 (olyaer_input 상태로)
        olayer_inputs = self.reverse_activation_function(olayer_outputs)
        

        # olayer > hlayer2
        hlayer2_outputs = np.dot(self.w_h2_o.T, olayer_inputs)
        # 0.01 - 0.99 의 값으로 정규화
        hlayer2_outputs -= np.min(hlayer2_outputs)
        hlayer2_outputs /= np.max(hlayer2_outputs)
        hlayer2_outputs *= 0.98
        hlayer2_outputs += 0.01        
        # 시그모이드 이전으로 되돌림
        hlayer2_inputs = self.reverse_activation_function(hlayer2_outputs)


        # hlayer2 > hlayer1
        hlayer1_outputs = np.dot(self.w_h1_h2.T, hlayer2_inputs)
        # 0.01 - 0.99 의 값으로 정규화
        hlayer1_outputs -= np.min(hlayer1_outputs)
        hlayer1_outputs /= np.max(hlayer1_outputs)
        hlayer1_outputs *= 0.98
        hlayer1_outputs += 0.01        
        # 시그모이드 이전으로 되돌림
        hlayer1_inputs = self.reverse_activation_function(hlayer1_outputs)


        # input을 계산
        inputs = np.dot(self.w_i_h1.T, hlayer1_inputs)
        # 0.01 - 0.99 의 값으로 정규화
        inputs -= np.min(inputs)
        inputs /= np.max(inputs)
        inputs *= 0.98
        inputs += 0.01
        
        return inputs


# import시 실행되지 않음.
if __name__ == "__main__" :

    ### CONFIG ==================================================================
    input_nodes = 784
    hidden1_nodes = 100
    hidden2_nodes = 100
    output_nodes = 10
    lr = 0.3


    ### CREATE ==================================================================
    my_network = Neural_Network(input_nodes, hidden1_nodes, hidden2_nodes, output_nodes, lr)


    ### TRAIN ==================================================================
    data_file = open('mnist_train_100.csv', 'r')
    all_lines = data_file.readlines()
    data_file.close()

    idx = len(all_lines)
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