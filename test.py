import network
import json
import sys
import CrossEntropyCost
import numpy as np
import imageio
from PIL import Image
import PIL
import matplotlib.pyplot as plt


def loading(filesave):
    f = open(filesave,"r")
    data = json.load(f)
    cost = getattr(sys.modules[__name__],data["cost"])
    net = network.Network(data["sizes"],cost = cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net
def calculate_output(x):
    activation = x
    for b,w in zip(net.biases,net.weights):
        z = np.dot(w,activation)+b
        activation = net.sigmoid(z)
    return activation
def load_image(image_name,num_px=28):
    fname = "images/"+image_name
    image = np.asarray(Image.open(fname).resize((num_px,num_px),Image.ANTIALIAS))
    image = image[:,:,:1]
    image = image.reshape((num_px*num_px,1))/255.0
    return image
def predict(out_activation):
    return np.argmax(out_activation)

if __name__ == '__main__':
    net = loading("model.json")
    x = load_image("seven.jpg")
    output_activations = calculate_output(x)
    prediction = predict(output_activations)
    print(prediction)
