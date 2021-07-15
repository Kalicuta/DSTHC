import numpy as np
import matplotlib.pyplot as plt

from hw1_logistic import *
from hw1_logistic import predict as predict_logistic
from hw2_planar_data import *
from utils.icehot_utils import get_image_online, plot_decision_boundary


class TestLogistic(object):
    def __init__(self):
        self.train_set_x = None
        self.train_set_y = None
        self.test_set_x  = None
        self.test_set_y  = None
        self.model       = None
        self.num_px      = 64
        self.classes     = None 
        
    def load_data(self):
        train_set_x_orig, self.train_set_y, test_set_x_orig, self.test_set_y, self.classes = load_dataset()
        # plt.imshow(train_set_x_orig[25])
        # plt.show()
    
        # train_set_x_orig is a numpy-array of shape (m_train, num_px, num_px, 3). 
        m_train = self.train_set_y.shape[0]
        m_test = self.test_set_y.shape[0]
        self.num_px = train_set_x_orig.shape[1]
        
        print(self.train_set_y.shape)
        print(self.test_set_y.shape)
        print(train_set_x_orig.shape)
    
        # Reshape the training and test data sets so that images of size (num_px, num_px, 3) 
        # are flattened into single vectors of shape (num_px  ∗ num_px  ∗ 3, 1).
        # X_flatten = X.reshape(X.shape[0], -1).T      
        train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
        test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
        
        self.train_set_x = train_set_x_flatten / 255
        self.test_set_x  = test_set_x_flatten / 255

    def train(self, num_iterations=2000, learning_rate=0.005, print_cost=True):
        self.model = model(self.train_set_x, self.train_set_y, self.test_set_x, self.test_set_y, 
                           num_iterations=num_iterations, learning_rate=learning_rate, print_cost=print_cost)

    def predict(self, url=None):
        if not url:
            my_image = "my_image.jpg"   
            fname = "images/" + my_image
            image = np.array(Image.open(fname).resize((self.num_px, self.num_px)))
        else:
            image = get_image_online(url, self.num_px)
        plt.imshow(image)
        image = image / 255.
        image = image.reshape((1, self.num_px * self.num_px * 3)).T
        my_predicted_image = predict_logistic(self.model["w"], self.model["b"], image)
        
        print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + self.classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
        plt.show()
    

class TestPlanarData(object):
    def load_planar_dataset(self):
        np.random.seed(1)
        m = 400 # number of examples
        N = int(m/2) # number of points per class
        D = 2 # dimensionality
        X = np.zeros((m,D)) # data matrix where each row is a single example
        Y = np.zeros((m,1), dtype='uint8') # labels vector (0 for red, 1 for blue)
        a = 4 # maximum ray of the flower
    
        for j in range(2):
            ix = range(N*j,N*(j+1))
            t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.2 # theta
            r = a*np.sin(4*t) + np.random.randn(N)*0.2 # radius
            X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
            Y[ix] = j
            
        X = X.T
        Y = Y.T
    
        return X, Y


def unit_test_logistic():
    test = TestLogistic()
    test.load_data()
    test.train()
    url = "https://www.petheaven.co.za/blog/wp-content/uploads/2014/08/The_Wolverine-_Cat1.jpg.pagespeed.ce.QwStveyyoF.jpg"
    test.predict(url)

def unit_test_planar_data():
    test = TestPlanarData()

    X, Y = test.load_planar_dataset()
    print(X.shape, Y.shape)
    plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral)
    plt.show()

    # parameters = nn_model(X, Y, n_h=4, num_iterations=10000, print_cost=True)
    # plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
    # plt.title("Decision Boundary for hidden layer size " + str(4))
    # plt.show()

    plt.figure(figsize=(16, 32))
    hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]

    for i, n_h in enumerate(hidden_layer_sizes):
        plt.subplot(5, 2, i+1)
        plt.title('Hidden Layer of size %d' % n_h)
        parameters = nn_model(X, Y, n_h, num_iterations = 5000)
        plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
        predictions = predict(parameters, X)
        accuracy = float((np.dot(Y,predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size)*100)
        print ("Accuracy for {} hidden units: {} %".format(n_h, accuracy))

    plt.show()

    
if __name__ == "__main__":
    unit_test_logistic()

    unit_test_planar_data()


