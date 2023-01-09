from sklearn.svm import *
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

# POLY_DEG_N = 10
# COEFF0_N = 5
# C_list = [0.01, 10, 10000]
# gamma_list = ['scale', 'auto']
# max_accuracy = 0
# max_parameters = ['linear', 0.01, None, None, None]

# for c in C_list:
#     print()
#     svm = SVC(kernel='linear', C=c)
    
#     digits_pred = svm.fit(digits_train, digits_labels_train).predict(digits_test)
    
#     acc_digit_linear = accuracy_score(digits_labels_test, digits_pred)*100

#     print(f'Accuracy of Linear at regularization parameter={c} : {acc_digit_linear} %')
#     if acc_digit_linear > max_accuracy:
#         max_accuracy = acc_digit_linear
#         max_parameters = ['linear', c, None, None, None]

#     print()

#     for c0 in range(0, COEFF0_N+1):
#         for g in gamma_list:
#             for d in range(1,POLY_DEG_N+1):
#                 svm = SVC(kernel='poly', C=c, degree=d, coef0=c0, gamma=g)

#                 digits_pred = svm.fit(digits_train, digits_labels_train).predict(digits_test)

#                 acc_digit_poly = accuracy_score(digits_labels_test, digits_pred)*100

#                 print(f'Accuracy of Polynomial at regularization parameter={c}, degree={d}, gamma={g}, coef0={c0} : {acc_digit_poly} %')
#                 if acc_digit_poly > max_accuracy:
#                     max_accuracy = acc_digit_poly
#                     max_parameters = ['poly', c, d, g, c0]

#     print()

#     for g in gamma_list:
#         svm = SVC(kernel='rbf', C=c, gamma=g)

#         digits_pred = svm.fit(digits_train, digits_labels_train).predict(digits_test)

#         acc_digit_rbf = accuracy_score(digits_labels_test, digits_pred)*100

#         print(f'Accuracy of RBF at regularization parameter={c}, gamma={g} : {acc_digit_rbf} %')
#         if acc_digit_rbf > max_accuracy:
#             max_accuracy = acc_digit_rbf
#             max_parameters = ['rbf', c, None, g, None]

#     print()

#     for c0 in range(0,COEFF0_N+1):
#         for g in gamma_list:
#             svm = SVC(kernel='sigmoid', C=c, coef0=c0, gamma=g)

#             digits_pred = svm.fit(digits_train, digits_labels_train).predict(digits_test)

#             acc_digit_sigmoid = accuracy_score(digits_labels_test, digits_pred)*100

#             print(f'Accuracy of Sigmoid at regularization parameter={c}, gamma={g}, coef0={c0} : {acc_digit_sigmoid} %')
#             if acc_digit_sigmoid > max_accuracy:
#                 max_accuracy = acc_digit_sigmoid
#                 max_parameters = ['sigmoid', c, None, g, c0]

# print('\n\n')
# if max_parameters[3] == None:
#     print(f'Maximum Accuracy is {max_accuracy} at kernel={max_parameters[0]}, c={max_parameters[1]}')
# elif max_parameters[4] == None:
#     print(f'Maximum Accuracy is {max_accuracy} at kernel={max_parameters[0]}, c={max_parameters[1]}, gamma={max_parameters[3]}')
# elif max_parameters[2] == None:
#     print(f'Maximum Accuracy is {max_accuracy} at kernel={max_parameters[0]}, c={max_parameters[1]}, gamma={max_parameters[3]}, coef0={max_parameters[4]}')
# else:
#     print(f'Maximum Accuracy is {max_accuracy} at kernel={max_parameters[0]}, c={max_parameters[1]}, degree={max_parameters[2]}, gamma={max_parameters[3]}, coef0={max_parameters[4]}')


print()
optimal = ['rbf', 10, 'scale']
svm = SVC(kernel=optimal[0], C=optimal[1], gamma=optimal[2])

digits_pred = svm.fit(digits_train, digits_labels_train).predict(digits_test)

acc_digit = accuracy_score(digits_labels_test, digits_pred)*100

print(f'Best Accuracy at kernel={optimal[0]}, regularization parameter={optimal[1]}, gamma={optimal[2]} : {acc_digit} %')
print()


print('//////////////////////////////////////////////////////////////////')
print('//                                                              //')
print('//                          Faces Data                          //')
print('//                                                              //')
print('//////////////////////////////////////////////////////////////////')
print()

FACE_TRAINING_SIZE = 451
FACE_TESTING_SIZE = 150

faces_labels_train, faces_labels_test, faces_train, faces_test = loadFacesData(FACE_TRAINING_SIZE, FACE_TESTING_SIZE)

# POLY_DEG_N = 10
# COEFF0_N = 5
# C_list = [0.01, 10, 10000]
# gamma_list = ['scale', 'auto']
# max_accuracy = 0
# max_parameters = ['linear', 0.01, None, None, None]

# for c in C_list:
#     print()
#     svm = SVC(kernel='linear', C=c)
    
#     faces_pred = svm.fit(faces_train, faces_labels_train).predict(faces_test)
    
#     acc_face_linear = accuracy_score(faces_labels_test, faces_pred)*100

#     print(f'Accuracy of Linear at regularization parameter={c} : {acc_face_linear} %')
#     if acc_face_linear > max_accuracy:
#         max_accuracy = acc_face_linear
#         max_parameters = ['linear', c, None, None, None]

#     print()

#     for c0 in range(0, COEFF0_N+1):
#         for g in gamma_list:
#             for d in range(1,POLY_DEG_N+1):
#                 svm = SVC(kernel='poly', C=c, degree=d, gamma=g, coef0=c0)

#                 faces_pred = svm.fit(faces_train, faces_labels_train).predict(faces_test)

#                 acc_face_poly = accuracy_score(faces_labels_test, faces_pred)*100

#                 print(f'Accuracy of Polynomial at regularization parameter={c}, degree={d}, gamma={g}, coef0={c0} : {acc_face_poly} %')
#                 if acc_face_poly > max_accuracy:
#                     max_accuracy = acc_face_poly
#                     max_parameters = ['poly', c, d, g, c0]

#     print()

#     for g in gamma_list:
#         svm = SVC(kernel='rbf', C=c, gamma=g)

#         faces_pred = svm.fit(faces_train, faces_labels_train).predict(faces_test)

#         acc_face_rbf = accuracy_score(faces_labels_test, faces_pred)*100

#         print(f'Accuracy of RBF at regularization parameter={c}, gamma={g} : {acc_face_rbf} %')
#         if acc_face_rbf > max_accuracy:
#             max_accuracy = acc_face_rbf
#             max_parameters = ['rbf', c, None, g, None]

#     print()

#     for c0 in range(0,COEFF0_N+1):
#         for g in gamma_list:
#             svm = SVC(kernel='sigmoid', C=c, gamma=g, coef0=c0)

#             faces_pred = svm.fit(faces_train, faces_labels_train).predict(faces_test)

#             acc_face_sigmoid = accuracy_score(faces_labels_test, faces_pred)*100

#             print(f'Accuracy of Sigmoid at regularization parameter={c}, gamma={g}, coef0={c0} : {acc_face_sigmoid} %')
#             if acc_face_sigmoid > max_accuracy:
#                 max_accuracy = acc_face_sigmoid
#                 max_parameters = ['sigmoid', c, None, g, c0]

# print('\n\n')
# if max_parameters[3] == None:
#     print(f'Maximum Accuracy is {max_accuracy} at kernel={max_parameters[0]}, c={max_parameters[1]}')
# elif max_parameters[4] == None:
#     print(f'Maximum Accuracy is {max_accuracy} at kernel={max_parameters[0]}, c={max_parameters[1]}, gamma={max_parameters[3]}')
# elif max_parameters[2] == None:
#     print(f'Maximum Accuracy is {max_accuracy} at kernel={max_parameters[0]}, c={max_parameters[1]}, gamma={max_parameters[3]}, coef0={max_parameters[4]}')
# else:
#     print(f'Maximum Accuracy is {max_accuracy} at kernel={max_parameters[0]}, c={max_parameters[1]}, degree={max_parameters[2]}, gamma={max_parameters[3]}, coef0={max_parameters[4]}')


print()
optimal = ['poly', 0.01, 8, 'scale', 1]
svm = SVC(kernel=optimal[0], C=optimal[1], degree=optimal[2], gamma=optimal[3], coef0=optimal[4])

faces_pred = svm.fit(faces_train, faces_labels_train).predict(faces_test)

acc_face = accuracy_score(faces_labels_test, faces_pred)*100

print(f'Best Accuracy at kernel={optimal[0]}, regularization parameter={optimal[1]}, degree={optimal[2]}, gamma={optimal[3]}, coef0={optimal[4]} : {acc_face} %')
print()