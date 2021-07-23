# подключаем библиотеку numpy
import numpy
# библиотека с сигмоидой
import scipy.special
import matplotlib.pyplot


# %matplotlib inline

# Определение класса нейронной сети
class neuralNetwork:
    # инициализация нейронной сети
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # задать количество узлов во входном, скрытом и выходном слое
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        # Матрицы весовых коэффициентов связей, wih и who.
        # Весовые коэффициенты связей между узлом i и узлом j
        # следующего слоя обозначены как w_i_j:
        # wll w21
        # wl2 w22 и т.д.
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        # коэффициент обучения
        self.lr = learningrate
        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    def train(self, inputs_list, targets_list):
        # преобразование списка входных значений
        # в двухмерный массив
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        # рассчитать входящие сигналы для скрытого слоя
        hidden_inputs = numpy.dot(self.wih, inputs)
        # рассчитать исходящие сигналы для скрытого слоя
        hidden_outputs = self.activation_function(hidden_inputs)
        # рассчитать входящие сигналы для выходного слоя
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # рассчитать исходящие сигналы для выходного слоя
        final_outputs = self.activation_function(final_inputs)
        # ошибки выходного слоя =
        # (целевое значение - фактическое значение)
        output_errors = targets - final_outputs
        # ошибки скрытого слоя - это ошибки output_errors,
        # распределенные пропорционально весовым коэффициентам связей
        # и рекомбинированные на скрытых узлах
        hidden_errors = numpy.dot(self.who.T, output_errors)
        # обновить весовые коэффициенты для связей между
        # скрытым и выходным слоями
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                        numpy.transpose(hidden_outputs))
        # обновить весовые коэффициенты для связей между
        # входным и скрытым слоями
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                        numpy.transpose(inputs))
        pass

    def query(self, inputs_list):
        # преобразовать список входных значений
        # в двухмерный массив
        inputs = numpy.array(inputs_list, ndmin=2).T
        # рассчитать входящие сигналы для скрытого слоя
        hidden_inputs = numpy.dot(self.wih, inputs)

        # рассчитать исходящие сигналы для скрытого слоя
        hidden_outputs = self.activation_function(hidden_inputs)
        # рассчитать входящие сигналы для выходного слоя
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # рассчитать исходящие сигналы для выходного слоя
        final_outputs = self.activation_function(final_inputs)
        return final_outputs


# количетсво входных скрытых узлов
input_nodes = 784
hidden_nodes = 10
output_nodes = 10
# коэффициент обучения
learning_grate = 0, 3
# Создаем экзэмпляр нейронной сети
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_grate)

# загрузить в список тестовый набор данных csv-файла набора MNIST
training_data_file = open(r"C:\Users\Nick Smirnov\PycharmProjects\neural\mnist_test.csv")
training_data_list = training_data_file.readlines()
training_data_file.close()
# тренировка нейронной сети
# перебрать все записи в тренировочном наборе
for record in training_data_list:
    # получить все значения используя символы запятой в качестве разделителя
    all_values = record.split(',')
    # масштабировать и сместить входные значения
    inputs = numpy.zeros(output_nodes) + 0.01
    # создать целевые маркерные значения равноы 0,01
    targets = numpy.zeros(output_nodes) + 0.01
    # all_values целевое маркерное значение для данной записи
    targets[int(all_values[0])] = 0.99
    n.train(inputs, targets)
    pass
