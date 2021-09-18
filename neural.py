import numpy as np
#import scipy.special
#  определение класса нейронной сети
class neuralNetwork:
    # инициализация
    def __init__(self, inputnodes, hiddennodes, outnodes, learningrate):
        # количество узлов во входном скрытом и выходном слое
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outnodes
        # коэффициент обучения
        self.lr = learningrate
        # Матрицы весовых коэффициентов связей wih ( между входным и скрытым
        # слоями) и who (между скрытым и выходным слоями).
        # Весовые коэффициенты связей между узлом i и узлом j следующего слоя # обозначены как w_i _j :
        # w11 w21 w31 w41 w51
        # w12 w22 w32 w42 w52
        # w13 w23 w33 w43 w53
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        # использование сигмоиды в качестве функции активации
        self.activation_function = lambda x: 1 / (1 + np.exp(-x))
        pass

    def train(self, inputs_list, targets_list):
        # преобразование списка входных значений в двухмерный массив
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        # рассчитать входящие сигналы для скрытого слоя
        hidden_inputs = np.dot(self.wih, inputs)
        # рассчитать исходящие сигналы для скрытого слоя
        hidden_outputs = self.activation_function(hidden_inputs)
        # рассчитать входящие сигналы для выходного слоя
        final_inputs = np.dot(self.who, hidden_outputs)
        # рассчитать исходящие сигналы для выходного слоя
        final_outputs = self.activation_function(final_inputs)
        # ошибки выходного слоя = ( целевое значение - фактическое значение)
        output_errors = targets - final_outputs
        # ошибки скрытого слоя - это ошибки output_errors, распределенные пропорционально весовым коэффициентам связей
        # и рекомбинированные на скрытых узлах
        # Ошибка_скрытый = Траспонированная_матрица_весов_скрытый_выходной * Ошибка_выходной
        hidden_errors = np.dot(self.who.T, output_errors)
        # обновить весовые коэффициенты для связей между скрытым и выходным слоями
        # Измененный_вес = Коэффиц.обучения * Ошибка_вых * Сигнал_вых * (1 -Сигн_вых) * Транспор_Сигнал_скрытогослоя
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
        # обновить весовые коэффициенты для связей между входным и скрытым слоями
        # Измененный_вес = Коэффиц.обучения * Ошибка_скрытСлоя * Сигнал_скрытСлоя * (1 -Сигн_скрытСлоя) * Транспор_Сигнал_входа
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))

        pass

    def query(self, inputs_list):
        # преобразовать список входных значений
        # в двухмерный массив
        inputs = np.array(inputs_list, ndmin=2).T
        # рассчитать входящие сигналы для скрытого слоя
        hidden_inputs = np.dot(self.wih, inputs)
        # рассчитать исходящие сигналы для скрытого слоя
        hidden_outputs = self.activation_function(hidden_inputs)
        # рассчитать входящие сиг налы для выходног о слоя
        final_inputs = np.dot(self.who, hidden_outputs)
        # рассчитать исходящие сигналы для выходног о слоя
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

# количество входных, скрытых и выходных узлов
input_nodes = 784
hidden_nodes = 128
output_nodes = 10

# коэффициент обучения равен 0, 3
learning_rate = 0.1
#создать экземпляр нейронной сети
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# загрузить в список тестовый набор данных CSV-файла набора MNIST
training_data_file = open("mnist_dataset/mnist_train_100.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# тренировка нейронной сети
# перебрать все записи в тренировочном наборе данных
epochs = 5
for e in range(epochs):
    for record in training_data_list:
        # получить список значений, используя символы запятой ( 1, 1) # в качестве разделителей
        all_values = record.split(',')
        # масштабировать и сместить входные значения
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # создать целевые выходные значения (все равны 0 01, за исключением  желаемого маркерного значения, равного 0,99)
        targets = np.zeros(output_nodes) + 0.01
        # all_values[0] - целевое маркерное значение для данной записи
        targets[int(all_values[0])] =0.99
        n.train(inputs, targets)
    pass
pass

# загрузить в список тестовый набор данных CSV-файла набора MNIST
test_data_file = open("mnist_dataset/mnist_test_10.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

 # тестирование нейронной сети
# журнал оценок работы сети, первоначально пустой
scorecard = []
# перебрать все записи в тестовом наборе данных
for record in test_data_list:
# получить список значений из записи, используя символы  запятой (*,1) в качестве разделителей
    all_values = record.split(',')
    # правильный ответ - первое значение
    correct_label = int(all_values[0])
    print(correct_label, "истинный маркер")
    # масштабировать и сместить входные значения
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01 # опрос сети
    outputs = n.query(inputs)
    # индекс наибольшего значения является маркерным значением
    label = np.argmax(outputs)
    print(label, "ответ сети")
    # присоединить оценку ответа сети к концу списка
    if (label == correct_label):
    # в случае правильного ответа сети присоединить  к списку значение 1
        scorecard.append(1)
    else:
    # в случае неправильного ответа сети присоединить  к списку значение 0
        scorecard.append(0)
    pass
pass

# рассчитать показатель эффективности в виде
# доли правильных ответов
scorecard_array = np.asarray(scorecard)
aff = scorecard_array.sum()/scorecard_array.size
print("эффективность = ", aff)


