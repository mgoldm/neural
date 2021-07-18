#подключаем библиотеку numpy
import numpy
#библиотека с сигмоидой 
import scipy.special
# Определение класса нейронной сети
class neuralNetwork:
    # инициализация нейронной сети
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # задать количество ухлов во входном,скрытом и выходном слое
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        # коэффициент обучения
        self.lr = learningrate
        #Матрица весовых коэффициентов связей wih и who
        #Весовые коэффициенты связей между узлом i и узлом j следующего слоя обозначены, как w_i_j
        self.wih=numpy.random.normal(0.0, pow(self.hnodes, -0,5), (self.hnodes, self. inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0, 5), (self.onodes, self.hnodes))
        #Использоваение сигмоиды в качестве функции активации
        self.activation_function = lambda x: scipy.special.exipt(x)
        pass

    def train(self):
        pass

    def query(self, inputs_list):
        #преобразовать список входных значений в двумерный массив
        inputs = numpy.array(inputs_list, ndmin=2)
        #расчитать входящие сигналы для скрытого поля
        hidden_inputs = numpy.dot(self.wih, inputs)
        #расчитать исходящие сигналы для скрытого поля
        hidden_outputs = self.activation_function(hidden_inputs)
        #расчитать входящие сигналы для выходного поля
        final_inputs = numpy.dot(self.who, hidden_outputs)
        #расчитать исходящие сигналы для выходного поля
        final_outputs = self.activation_function(final_inputs)
        return final_outputs
        pass


# количетсво входных скрытых узлов
input_nodes = 3
hidden_nodes = 3
output_nodes = 3
# коэффициент обучения
learning_grate = 3
#Создаем экзэмпляр нейронной сети
n = neuralNetwork(input_nodes, hidden_nodes,output_nodes, learning_grate)
