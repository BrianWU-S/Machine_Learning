from libsvm import svm, svmutil, commonutil
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.neural_network import MLPClassifier


def load_data(train_path, test_path):
    y_train, x_train = commonutil.svm_read_problem(train_path)
    y_test, x_test = commonutil.svm_read_problem(test_path)
    print("Data train size:", np.shape(x_train), np.shape(y_train))
    print("Data train example:", x_train[0], y_train[0])
    print("Data test size:", np.shape(x_test), np.shape(y_test))
    print("Data test example:", x_test[0], y_test[0])
    return x_train, y_train, x_test, y_test


def convert_data(x_train, y_train, x_test, y_test):
    """
    convert the libsvm data style to sklearn data style: X:(n_samples,n_features), Y: (n_samples,)
    key max: 123, key min: 1
    We convert to one-hot encoding
    """
    # get one hot vector size
    one_hot_len = max([max(i.keys()) for i in x_train if i])
    print("One hot vector length:", one_hot_len)
    sk_x_train = np.zeros(shape=(np.shape(x_train)[0], one_hot_len))
    sk_y_train = np.array(y_train)  # np.array() and np.asarray: the former copies object by default
    sk_x_test = np.zeros(shape=(np.shape(x_test)[0], one_hot_len))
    sk_y_test = np.array(y_test)
    for row in range(np.shape(x_train)[0]):
        d = x_train[row]
        for k in d.keys():
            sk_x_train[row, k - 1] = 1
    for row in range(np.shape(x_test)[0]):
        d = x_test[row]
        for k in d.keys():
            sk_x_test[row, k - 1] = 1
    print("Sklearn data shape:", np.shape(sk_x_train), np.shape(sk_y_train), np.shape(sk_x_test), np.shape(sk_y_test))
    return sk_x_train, sk_y_train, sk_x_test, sk_y_test


def svm_train_and_test():
    # train and test
    model = svmutil.svm_train(Y_train, X_train, '-s 0 -t 2 -c 4 -h 0')  # -h 1: 69s ; -h 0: 67s
    print("Model Hyper-Parameters:\n", model.param)
    p_label, p_acc, p_val = svmutil.svm_predict(Y_test, X_test, model)
    print('\n', "Predicted Labels:", np.array(p_label))
    print("True Labels:", Y_test)
    print(' Probability estimate Values:', p_val)


def svm_hyper_parameters_tuning(X_train,Y_train,X_test,Y_test):
    # C_list = np.linspace(0.01, 0.1, 100)
    # Tick_list = np.round(np.linspace(0.01, 0.11, 26), decimals=2)
    # C_list = np.linspace(0.1, 5.0, 50)
    # Tick_list = np.linspace(0.1, 5.0, 26)
    C_list = np.linspace(30.0, 55.0, 50)
    Tick_list = np.linspace(30.0, 55.0, 26)
    
    acc_list = []  # accuracy (for classification)
    mse_list = []  # mean squared error
    scc_list = []  # squared correlation coefficient
    for C_value in C_list:
        model = svmutil.svm_train(Y_train, X_train, '-s 0 -t 2 -c %f -h 0' % C_value)
        p_label, p_acc, p_val = svmutil.svm_predict(Y_test, X_test, model)
        print("C:", C_value, " Acc:", p_acc)
        acc_list.append(p_acc[0])
        mse_list.append(p_acc[1])
        scc_list.append(p_acc[2])
    
    plt.figure(figsize=(20, 10))
    plt.plot(C_list, acc_list)
    plt.xticks(Tick_list, fontsize=15)
    plt.yticks(fontsize=18)
    plt.xlabel("Regularization Hyper parameter C", fontdict={'size': 18})
    plt.ylabel("Accuracy", fontdict={'size': 18})
    plt.show()
    
    plt.figure(figsize=(20, 10))
    plt.plot(C_list, mse_list)
    plt.xticks(Tick_list, fontsize=15)
    plt.yticks(fontsize=18)
    plt.xlabel("Regularization Hyper parameter C", fontdict={'size': 18})
    plt.ylabel("Mean Squared Error", fontdict={'size': 18})
    plt.show()
    
    plt.figure(figsize=(20, 10))
    plt.plot(C_list, scc_list)
    plt.xticks(Tick_list, fontsize=15)
    plt.yticks(fontsize=18)
    plt.xlabel("Regularization Hyper parameter C", fontdict={'size': 18})
    plt.ylabel("Squared Correlation Coefficient", fontdict={'size': 18})
    plt.show()


def mlp_train_and_test():
    """
     Note: in adam, lbfgs, max_iter is the number of epochs, not max_iterations
     Max iter: 15000    --> 确保网络能训练到收敛，若训练集上精度不高则是high bias，需要大的网络规模，若是test set 上精度不高，则是high
     variance， 此时网络已经够大了，需要调整epoch
    """
    # clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(128, 64, 32), random_state=1, max_iter=300,
    #                     verbose=False, activation='relu',batch_size=64)
    clf = MLPClassifier(solver='adam', hidden_layer_sizes=(128, 4), random_state=1, max_iter=500,
                        verbose=False, activation='relu', batch_size=128, learning_rate_init=0.0001)
    clf.fit(X=sk_X_train, y=sk_Y_train)
    print("clf train score:", clf.score(sk_X_train, sk_Y_train))
    print("clf test score:", clf.score(sk_X_test, sk_Y_test))


def mlp_hyper_parameters_tuning(sk_X_train, sk_Y_train, sk_X_test, sk_Y_test):
    Epoch_list = np.arange(5, 100, 5)
    first_layer_units = [8, 16, 16, 32, 32, 64, 64, 128, 128, 128]
    second_layer_units = [2, 4, 8, 4, 8, 4, 8, 2, 4, 8]
    epoch_train_acc_list = []  # accuracy (for classification)
    epoch_test_acc_list = []
    net_train_acc_list = []  # accuracy (for classification)
    net_test_acc_list = []
    for i in range(len(first_layer_units)):
        print("L1:", first_layer_units[i], "L2:", second_layer_units[i])
        clf = MLPClassifier(solver='adam', hidden_layer_sizes=(first_layer_units[i], second_layer_units[i]),
                            random_state=1, max_iter=5000,
                            verbose=False, activation='relu', batch_size=128, learning_rate_init=0.0001)
        clf.fit(X=sk_X_train, y=sk_Y_train)
        train_acc = clf.score(sk_X_train, sk_Y_train)
        test_acc = clf.score(sk_X_test, sk_Y_test)
        print("clf train score:", train_acc)
        print("clf test score:", test_acc)
        net_train_acc_list.append(train_acc)
        net_test_acc_list.append(test_acc)
    
    for epoch in Epoch_list:
        print("Epoch:", epoch)
        clf = MLPClassifier(solver='adam', hidden_layer_sizes=(64, 4), random_state=1, max_iter=int(epoch),
                            verbose=False, activation='relu', batch_size=128, learning_rate_init=0.0001)
        clf.fit(X=sk_X_train, y=sk_Y_train)
        train_acc = clf.score(sk_X_train, sk_Y_train)
        test_acc = clf.score(sk_X_test, sk_Y_test)
        print("clf train score:", train_acc)
        print("clf test score:", test_acc)
        epoch_train_acc_list.append(train_acc)
        epoch_test_acc_list.append(test_acc)
    
    # Network size
    plt.figure(figsize=(16, 8))
    plt.plot(np.arange(len(net_train_acc_list)), net_train_acc_list)
    plt.plot(np.arange(len(net_test_acc_list)), net_test_acc_list)
    plt.xticks(ticks=np.arange(len(first_layer_units)),
               labels=[("L1:" + str(first_layer_units[j]) + " L2:" + str(second_layer_units[j])) for j in
                       range(len(first_layer_units))], fontsize=12)
    plt.yticks(fontsize=18)
    plt.xlabel("Settings", fontdict={'size': 18})
    plt.ylabel("Accuracy", fontdict={'size': 18})
    plt.title("Hyper parameter network size", fontdict={'size': 18})
    plt.legend(["Train acc", "Test acc"], fontsize=18)
    plt.show()
    
    # Epoches
    plt.figure(figsize=(16, 8))
    plt.plot(Epoch_list, epoch_train_acc_list)
    plt.plot(Epoch_list, epoch_test_acc_list)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=18)
    plt.xlabel("Epoches", fontdict={'size': 18})
    plt.ylabel("Accuracy", fontdict={'size': 18})
    plt.title("Hyper parameter epoch graph", fontdict={'size': 18})
    plt.legend(["Train acc", "Test acc"], fontsize=18)
    plt.show()


if __name__ == '__main__':
    w6a_train_path = r'D:\Google_Download\Machine_Learning\Assignments\Assignment3\Classifications_SVM_MLP\data\w6a\w6a_train.txt'
    w6a_test_path = r'D:\Google_Download\Machine_Learning\Assignments\Assignment3\Classifications_SVM_MLP\data\w6a\w6a_test.txt'
    a9a_train_path = r'D:\Google_Download\Machine_Learning\Assignments\Assignment3\Classifications_SVM_MLP\data\a9a\a9a_train.txt'
    a9a_test_path = r'D:\Google_Download\Machine_Learning\Assignments\Assignment3\Classifications_SVM_MLP\data\a9a\a9a_test.txt'
    t1 = time.time()
    
    # a9a dataset
    X_train, Y_train, X_test, Y_test = load_data(a9a_train_path, a9a_test_path)
    sk_X_train, sk_Y_train, sk_X_test, sk_Y_test = convert_data(X_train, Y_train, X_test, Y_test)
    svm_train_and_test()
    svm_hyper_parameters_tuning(X_train, Y_train, X_test, Y_test)
    mlp_train_and_test()
    mlp_hyper_parameters_tuning(sk_X_train, sk_Y_train, sk_X_test, sk_Y_test)
    
    # w6a dataset
    X_train_2, Y_train_2, X_test_2, Y_test_2 = load_data(w6a_train_path, w6a_test_path)
    sk_X_train_2, sk_Y_train_2, sk_X_test_2, sk_Y_test_2 = convert_data(X_train_2, Y_train_2, X_test_2, Y_test_2)
    svm_train_and_test()
    svm_hyper_parameters_tuning(X_train_2, Y_train_2, X_test_2, Y_test_2)
    mlp_train_and_test()
    mlp_hyper_parameters_tuning(sk_X_train_2, sk_Y_train_2, sk_X_test_2, sk_Y_test_2)
    
    print("Training time:", time.time() - t1)
