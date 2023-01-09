from sklearn.neural_network import *
from sklearn.metrics import accuracy_score
from Data_Load import *

print('///////////////////////////////////////////////////////////////////')
print('//                                                               //')
print('//                          Digits Data                          //')
print('//                                                               //')
print('///////////////////////////////////////////////////////////////////')

DIGIT_TRAINING_SIZE = 5000
DIGIT_TESTING_SIZE = 1000

digits_labels_train, digits_labels_test, digits_train, digits_test = loadDigitsData(DIGIT_TRAINING_SIZE, DIGIT_TESTING_SIZE)

activation_functions = ['identity', 'logistic', 'tanh', 'relu']
learning_rates = [1, 0.1, 0.01, 0.001]
max_accuracy = 0
max_data = [1, 50, 'identity', 1]
for h in range(50, 300, 50):
    print()
    for a in activation_functions:
        print()
        for l in learning_rates:
            mlp = MLPClassifier(
                learning_rate_init=l,
                hidden_layer_sizes=(h,),
                activation=a,
                max_iter=1000
            )
            digits_pred = mlp.fit(digits_train, digits_labels_train).predict(digits_test)
            acc_digit = accuracy_score(digits_labels_test, digits_pred)*100
            print(f'Accuracy at 1 layer of {h} hidden neurons, {a} function, {l} learning rate : {round(acc_digit, 2)} %')
            if acc_digit>max_accuracy:
                max_accuracy = acc_digit
                max_data = [1, h, a, l]
    print()
    for a in activation_functions:
        print()
        for l in learning_rates:
            mlp = MLPClassifier(
                learning_rate_init=l,
                hidden_layer_sizes=(h,h,),
                activation=a,
                max_iter=1000
            )
            digits_pred = mlp.fit(digits_train, digits_labels_train).predict(digits_test)
            acc_digit = accuracy_score(digits_labels_test, digits_pred)*100
            print(f'Accuracy at 2 layers of {h} hidden neurons, {a} function, {l} learning rate : {round(acc_digit, 2)} %')
            if acc_digit>max_accuracy:
                max_accuracy = acc_digit
                max_data = [2, h, a, l]
    print()
    for a in activation_functions:
        print()
        for l in learning_rates:
            mlp = MLPClassifier(
                learning_rate_init=l,
                hidden_layer_sizes=(h,h,h,),
                activation=a,
                max_iter=1000
            )
            digits_pred = mlp.fit(digits_train, digits_labels_train).predict(digits_test)
            acc_digit = accuracy_score(digits_labels_test, digits_pred)*100
            print(f'Accuracy at 3 layers of {h} hidden neurons, {a} function, {l} learning rate : {round(acc_digit, 2)} %')
            if acc_digit>max_accuracy:
                max_accuracy = acc_digit
                max_data = [3, h, a, l]

print()
print(f'Maximum Accuracy was {max_accuracy} % at {max_data[0]} layers of {max_data[1]} hidden neurons, {max_data[2]} function, {max_data[3]} learning rate')



print('//////////////////////////////////////////////////////////////////')
print('//                                                              //')
print('//                          Faces Data                          //')
print('//                                                              //')
print('//////////////////////////////////////////////////////////////////')
print()

FACE_TRAINING_SIZE = 451
FACE_TESTING_SIZE = 150

faces_labels_train, faces_labels_test, faces_train, faces_test = loadFacesData(FACE_TRAINING_SIZE, FACE_TESTING_SIZE)

activation_functions = ['identity', 'logistic', 'tanh', 'relu']
learning_rates = [1, 0.1, 0.01, 0.001]
max_accuracy = 0
max_data = [1, 50, 'identity', 1]
for h in range(100, 500, 100):
    print()
    for a in activation_functions:
        print()
        for l in learning_rates:
            mlp = MLPClassifier(
                learning_rate_init=l,
                hidden_layer_sizes=(h,),
                activation=a,
                max_iter=1000
            )
            faces_pred = mlp.fit(faces_train, faces_labels_train).predict(faces_test)
            acc_faces = accuracy_score(faces_labels_test, faces_pred)*100
            print(f'Accuracy at 1 layer of {h} hidden neurons, {a} function, {l} learning rate : {round(acc_faces, 2)} %')
            if acc_faces>max_accuracy:
                max_accuracy = acc_faces
                max_data = [1, h, a, l]
    print()
    for a in activation_functions:
        print()
        for l in learning_rates:
            mlp = MLPClassifier(
                learning_rate_init=l,
                hidden_layer_sizes=(h,h,),
                activation=a,
                max_iter=1000,
            )
            faces_pred = mlp.fit(faces_train, faces_labels_train).predict(faces_test)
            acc_faces = accuracy_score(faces_labels_test, faces_pred)*100
            print(f'Accuracy at 2 layers of {h} hidden neurons, {a} function, {l} learning rate : {round(acc_faces, 2)} %')
            if acc_faces>max_accuracy:
                max_accuracy = acc_faces
                max_data = [2, h, a, l]
    print()
    for a in activation_functions:
        print()
        for l in learning_rates:
            mlp = MLPClassifier(
                learning_rate_init=l,
                hidden_layer_sizes=(h,h,h,),
                activation=a,
                max_iter=1000
            )
            faces_pred = mlp.fit(faces_train, faces_labels_train).predict(faces_test)
            acc_faces = accuracy_score(faces_labels_test, faces_pred)*100
            print(f'Accuracy at 3 layers of {h} hidden neurons, {a} function, {l} learning rate : {round(acc_faces, 2)} %')
            if acc_faces>max_accuracy:
                max_accuracy = acc_faces
                max_data = [3, h, a, l]

print()
print(f'Maximum Accuracy was {max_accuracy} % at {max_data[0]} layers of {max_data[1]} hidden neurons, {max_data[2]} function, {max_data[3]} learning rate')