import mnist
from sklearn.svm import LinearSVC
import pickle
from matplotlib import pyplot as plt

x_train = mnist.train_images() #preparing for training
y_train = mnist.train_labels()
x_test = mnist.test_images() #preparing for testing
y_test = mnist.test_labels()

n_samples, nx, ny = x_train.shape
n_samples_test, nx_test, ny_test = x_test.shape
x_train = x_train.reshape((n_samples, nx*ny)) #reshape the array to 2 dimensions
x_test0 = x_test.reshape((n_samples_test, nx_test*ny_test))

#svm = LinearSVC()
#svm.fit(x_train, y_train)
#model_file = open("mnist.pickle", "wb") #storing training data
#pickle.dump(svm, model_file)
#acc = svm.score(x_train, y_train)
#print(acc)

model_file = open("mnist.pickle", "rb") #using traiing data
svm = pickle.load(model_file)
prediction = svm.predict(x_test0)

for i in range(len(prediction)): #this is where predictions happen
    plt.imshow(x_test[i])
    print("predicting: ", prediction[i])
    plt.show()