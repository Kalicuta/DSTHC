import numpy as np
import copy
import matplotlib.pyplot as plt
from PIL import Image

from lr_utils import load_dataset

# build a simple image-recognition algorithm that can correctly classify pictures as cat or non-cat.


def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s

def initialize_with_zeros(dim):
    w = np.zeros(shape = (dim, 1))
    b = 0.0
    return w, b

def propagate(w, b, X, Y):
    m = X.shape[1]

    A = sigmoid(np.dot(w.T, X) + b) 
    cost = (- 1 /  m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))

    dw = (1/m) * np.dot(X, (A - Y).T)
    db = (1/m) * np.sum(A - Y)

    cost  = np.squeeze(np.array(cost))
    grads = {"dw": dw, 
             "db": db}

    return grads, cost
 
def optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False):
    w = copy.deepcopy(w)
    b = copy.deepcopy(b)

    costs = []

    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print ("Cost after iteration %i: %f" %(i, cost))

    params = {"w": w, 
              "b": b}

    grads = {"dw": dw, 
             "db": db}

    return params, grads, costs

def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    A = sigmoid(np.dot(w.T, X) + b)
    
    for i in range(A.shape[1]):
        if A[0, i] > 0.5:
            Y_prediction[0, i] = 1

        else:
            Y_prediction[0, i] = 0

    return Y_prediction 

def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    w, b = initialize_with_zeros(X_train.shape[0])
    params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    w = params["w"]
    b = params["b"]

    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    if print_cost:
        print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {
        "costs": costs, 
        "Y_prediction_test":    Y_prediction_test,
        "Y_prediction_train":   Y_prediction_train,  
        "w":                    w,
        "b":                    b, 
        "learning_rate":        learning_rate, 
        "num_iterations":       num_iterations
    }

    return d

def get_image_online(url):
    from PIL import Image
    import requests
    from io import BytesIO

    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    image = np.array(img.resize((num_px, num_px)))
    return image

   
if __name__ == "__main__":
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
    # plt.imshow(train_set_x_orig[25])
    # plt.show()

    # train_set_x_orig is a numpy-array of shape (m_train, num_px, num_px, 3). 
    m_train = train_set_y.shape[0]
    m_test = test_set_y.shape[0]
    num_px = train_set_x_orig.shape[1]
    
    print(train_set_y.shape)
    print(test_set_y.shape)
    print(train_set_x_orig.shape)

    # Reshape the training and test data sets so that images of size (num_px, num_px, 3) 
    # are flattened into single vectors of shape (num_px  ∗ num_px  ∗ 3, 1).
    # X_flatten = X.reshape(X.shape[0], -1).T      
    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
    
    train_set_x = train_set_x_flatten / 255
    test_set_x  = test_set_x_flatten / 255


    # My Test
    logistic_regression_model = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate=0.005, print_cost=True)
    
    # my_image = "my_image.jpg"   
    # fname = "images/" + my_image
    # image = np.array(Image.open(fname).resize((num_px, num_px)))

    url = "https://www.petheaven.co.za/blog/wp-content/uploads/2014/08/The_Wolverine-_Cat1.jpg.pagespeed.ce.QwStveyyoF.jpg"
    image = get_image_online(url)
    plt.imshow(image)
    image = image / 255.
    image = image.reshape((1, num_px * num_px * 3)).T
    my_predicted_image = predict(logistic_regression_model["w"], logistic_regression_model["b"], image)
    
    plt.show()

    print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")

