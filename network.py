import numpy as np
import random
import json
import sys
import QuadraticCost
import CrossEntropyCost
# Network initialization code assumes that the
# first layer of neurons is an input layer, and omits to set any biases for those neurons, since
# biases are only ever used in computing the outputs from later layers.
class Network(object):
    def __init__(self,sizes,cost = CrossEntropyCost.CrossEntropyCost):
            self.num_layers = len(sizes)
            self.sizes = sizes
            self.cost = cost
            self.default_weight_initializer()

    def default_weight_initializer(self):
            self.biases = [np.random.randn(y,1) for y in self.sizes[1:]]
            self.weights = [np.random.randn(y,x)/np.sqrt(x) for x,y in zip(self.sizes[:-1],self.sizes[1:])]

    def sigmoid(self,z):
        return 1.0/(1.0+np.exp(-z))

    def feedforward(self,a):
        for b,w in zip(self.biases,self.weights):
            a = self.sigmoid(np.dot(w,a)+b)
        return a

    def SGD(self,training_data, epochs, mini_batch_size,eta,lmbda = 0.0 ,evaluation_data =None , monitor_evaluation_cost =False ,monitor_evaluation_accuracy =False , monitor_training_cost =False ,monitor_training_accuracy = False ):
        training_data = list(training_data)
        n = len(training_data)
        if evaluation_data:
            evaluation_data = list(evaluation_data)
            n_data = len(evaluation_data)
        evaluation_cost, evaluation_accuracy = [],[]
        training_cost, training_accuracy = [],[]
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0,n,mini_batch_size)]
            for mini_batch in mini_batches:
                    self.update_mini_batch(mini_batch,eta,lmbda,len(training_data))
            print("Epoch %d Training completed"%j)
        if monitor_training_cost:
            cost1 = self.total_cost(training_data,lmbda)
            training_cost.append(cost1)
            print("Training cost: {}".format(cost1))
        if monitor_training_accuracy:
            acc = self.accuracy(training_data,convert=True)
            training_accuracy.append(acc)
            print("Training accuracy: {}".format(acc))
        if monitor_evaluation_cost:
            cost1 = self.total_cost(evaluation_data,lmbda,convert=True)
            evaluation_cost.append(cost1)
            print("Evaluation cost: {}".format(cost1))
        if monitor_evaluation_accuracy:
            acc = self.accuracy(evaluation_data)
            evaluation_accuracy.append(acc)
            print("Evaluation accuracy: {}".format(acc))
        print("")
        return evaluation_cost,evaluation_accuracy,training_cost,training_accuracy

    def update_mini_batch(self , mini_batch , eta, lmbda, n):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
                delta_nabla_b , delta_nabla_w = self.backprop(x, y)
                nabla_b = [nb+dnb for nb, dnb in zip(nabla_b , delta_nabla_b)]
                nabla_w = [nw+dnw for nw, dnw in zip(nabla_w , delta_nabla_w)]
        self.weights = [(1-eta*(lmbda/n))*w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights , nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases , nabla_b)]

    def backprop(self,x,y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        #feedforward
        activation = x #input activation layer
        activations = [x] #list to store all the activations layer by layer
        zs = [] #list to store all the z vectors layer by layer
        for b,w in zip(self.biases,self.weights):
                z = np.dot(w,activation)+b
                zs.append(z)
                activation = self.sigmoid(z)
                activations.append(activation)
        #backword pass
        delta = self.cost.delta(zs[-1],activations[-1],y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta,activations[-2].transpose())
        for l in range(2,self.num_layers):
            z = zs[-l]
            sp = self.sigmoid_prime(z)
            delta = np.multiply(np.dot(self.weights[-l+1].transpose(),delta),sp)
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta,activations[-l-1].transpose())
        return (nabla_b,nabla_w)

    def accuracy(self,data,convert=False):
        if convert:
            results = [(np.argmax(self.feedforward(x)),np.argmax(y)) for (x,y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)),y) for (x,y) in data]
        return sum(int(x==y) for (x,y) in results)

    def total_cost(self,data,lmbda,convert=False):
        cost = 0.0
        for x,y in data:
            a = self.feedforward(x)
        if convert:
            y = self.vectorized_y(y)
        cost += self.cost.fn(a,y)/len(data)
        cost += 0.5*(lmbda/len(data))*sum(np.linalg.norm(w)**2 for w in self.weights)
        return cost
    def vectorized_y(self,y):
        e = np.zeros((10,1))
        e[y] = 1.0
        return e
    def save(self,filesave):
        data_save = {"sizes":self.sizes,"weights":[w.tolist() for w in self.weights],"biases":[b.tolist() for b in self.biases],"cost":str(self.cost.__name__)}
        f = open(filesave,"w")
        json.dump(data_save,f)
        f.close()

    def cost_derivative(output_activation,y):
        return (output_activation-y)#return the partial derivation for the cost function(refer notes)

    def sigmoid_prime(self,z):
        return self.sigmoid(z)*(1-self.sigmoid(z))


