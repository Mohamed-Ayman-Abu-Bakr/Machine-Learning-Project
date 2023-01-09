from sklearn.svm import *
from sklearn.metrics import accuracy_score
from Data_Load import *

print('///////////////////////////////////////////////////////////////////')
print('//                                                               //')
print('//                          Digits Data                          //')
print('//                                                               //')
print('///////////////////////////////////////////////////////////////////')
print()

POLY_DEG_N = 10
DIGIT_TRAINING_SIZE = 5000
DIGIT_TESTING_SIZE = 1000

digits_labels_train, digits_labels_test, digits_train, digits_test = loadDigitsData(DIGIT_TRAINING_SIZE, DIGIT_TESTING_SIZE)

svm1 = SVC(kernel='linear')
svm2 = SVC(kernel='rbf')
svm3 = SVC(kernel='sigmoid')

digits_pred1 = svm1.fit(digits_train, digits_labels_train).predict(digits_test)
digits_pred2 = svm2.fit(digits_train, digits_labels_train).predict(digits_test)
digits_pred3 = svm3.fit(digits_train, digits_labels_train).predict(digits_test)

acc_digit_linear = accuracy_score(digits_labels_test, digits_pred1)*100
acc_digit_rbf = accuracy_score(digits_labels_test, digits_pred2)*100
acc_digit_sigmoid = accuracy_score(digits_labels_test, digits_pred3)*100

digits = {
    "Linear" : acc_digit_linear,
    "RBF" : acc_digit_rbf,
    "Sigmoid" : acc_digit_sigmoid,
}

for key, value in digits.items():
    print(f'Accuracy of {key} : {value} %')


for d in range(1,POLY_DEG_N+1):
    svm = SVC(kernel='poly', degree=d)

    digits_pred = svm.fit(digits_train, digits_labels_train).predict(digits_test)

    acc_digit_poly = accuracy_score(digits_labels_test, digits_pred)*100

    print(f'Accuracy of Polynomial at degree={d} : {acc_digit_poly} %')

print()



# print('//////////////////////////////////////////////////////////////////')
# print('//                                                              //')
# print('//                          Faces Data                          //')
# print('//                                                              //')
# print('//////////////////////////////////////////////////////////////////')
# print()

# FACE_TRAINING_SIZE = 451
# FACE_TESTING_SIZE = 150

# faces_labels_train, faces_labels_test, faces_train, faces_test = loadFacesData(FACE_TRAINING_SIZE, FACE_TESTING_SIZE)

