from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from Data_Load import *

import time
start_time = time.time()

print('///////////////////////////////////////////////////////////////////')
print('//                                                               //')
print('//                          Digits Data                          //')
print('//                                                               //')
print('///////////////////////////////////////////////////////////////////')
print()

K_N_DIGITS = 100
DIGIT_TRAINING_SIZE = 5000
DIGIT_TESTING_SIZE = 1000

digits_time = time.time()

digits_labels_train, digits_labels_test, digits_train, digits_test = loadDigitsData(DIGIT_TRAINING_SIZE, DIGIT_TESTING_SIZE)


euclidean_max_acc_digit = 0
manhattan_max_acc_digit = 0
euclidean_max_k_digit = 0
manhattan_max_k_digit = 0

k_arr = []
accuracy_manhattan = []
accuracy_euclidean = []
for k in range(1,K_N_DIGITS+1):
    temp_time = time.time()
    knn_manhattan = KNeighborsClassifier(n_neighbors=k, weights='distance', p=1)
    knn_euclidean = KNeighborsClassifier(n_neighbors=k, weights='distance', p=2)

    digits_pred1 = knn_manhattan.fit(digits_train, digits_labels_train).predict(digits_test)
    digits_pred2 = knn_euclidean.fit(digits_train, digits_labels_train).predict(digits_test)

    k_arr.append(k)
    accuracy_manhattan.append(accuracy_score(digits_labels_test, digits_pred1)*100)
    accuracy_euclidean.append(accuracy_score(digits_labels_test, digits_pred2)*100)

    print(f'Accuracy of Manhattan at k={k} : {accuracy_manhattan[-1]} %')
    print(f'Accuracy of Euclidean at k={k} : {accuracy_euclidean[-1]} %')
    print("Execution Time: %s seconds" % (time.time() - temp_time))
    print()
    if(accuracy_euclidean[-1]>euclidean_max_acc_digit):
        euclidean_max_acc_digit = accuracy_euclidean[-1]
        euclidean_max_k_digit = k
    if(accuracy_manhattan[-1]>manhattan_max_acc_digit):
        manhattan_max_acc_digit = accuracy_manhattan[-1]
        manhattan_max_k_digit = k

plt.figure('K Nearest Neigbors Classifier',figsize=(12, 5),)

plt.subplot(221)
plt.plot(k_arr, accuracy_manhattan)
plt.title('Manhattan Distance for Digits')
plt.xlabel('K (Number of Nearest Neighbors)')
plt.ylabel('Accuracy (%)')
plt.grid()
plt.locator_params(axis='x', nbins=20)

plt.subplot(222)
plt.plot(k_arr, accuracy_euclidean)
plt.title('Euclidean Distance for Digits')
plt.xlabel('K (Number of Nearest Neighbors)')
plt.ylabel('Accuracy (%)')
plt.grid('y')
plt.locator_params(axis='x', nbins=20)

print("Total Digits Execution Time: %s seconds" % (time.time() - digits_time))


print('//////////////////////////////////////////////////////////////////')
print('//                                                              //')
print('//                          Faces Data                          //')
print('//                                                              //')
print('//////////////////////////////////////////////////////////////////')
print()

K_N_FACES = 150
FACE_TRAINING_SIZE = 451
FACE_TESTING_SIZE = 150

faces_time = time.time()

faces_labels_train, faces_labels_test, faces_train, faces_test = loadFacesData(FACE_TRAINING_SIZE, FACE_TESTING_SIZE)


k_arr = []
accuracy_manhattan = []
accuracy_euclidean = []
euclidean_max_acc_face = 0
manhattan_max_acc_face = 0
euclidean_max_k_face = 0
manhattan_max_k_face = 0
for k in range(1,K_N_FACES+1):
    temp_time = time.time()
    knn_manhattan = KNeighborsClassifier(n_neighbors=k, weights='distance', p=1)
    knn_euclidean = KNeighborsClassifier(n_neighbors=k, weights='distance', p=2)

    faces_pred1 = knn_manhattan.fit(faces_train, faces_labels_train).predict(faces_test)
    faces_pred2 = knn_euclidean.fit(faces_train, faces_labels_train).predict(faces_test)

    k_arr.append(k)
    
    accuracy_manhattan.append(accuracy_score(faces_labels_test, faces_pred1)*100)
    accuracy_euclidean.append(accuracy_score(faces_labels_test, faces_pred2)*100)

    print(f'Accuracy of Manhattan at k={k} : {accuracy_manhattan[-1]} %')
    print(f'Accuracy of Euclidean at k={k} : {accuracy_euclidean[-1]} %')
    print("Execution Time: %s seconds" % (time.time() - temp_time))
    print()
    if(accuracy_euclidean[-1]>euclidean_max_acc_face):
        euclidean_max_acc_face = accuracy_euclidean[-1]
        euclidean_max_k_face = k
    if(accuracy_manhattan[-1]>manhattan_max_acc_face):
        manhattan_max_acc_face = accuracy_manhattan[-1]
        manhattan_max_k_face = k

print("Total Faces Execution Time: %s seconds" % (time.time() - faces_time))

plt.subplot(223)
plt.plot(k_arr, accuracy_manhattan)
plt.title('Manhattan Distance for Faces')
plt.xlabel('K (Number of Nearest Neighbors)')
plt.ylabel('Accuracy (%)')
plt.grid()
plt.locator_params(axis='x', nbins=20)

plt.subplot(224)
plt.plot(k_arr, accuracy_euclidean)
plt.title('Euclidean Distance for Faces')
plt.xlabel('K (Number of Nearest Neighbors)')
plt.ylabel('Accuracy (%)')
plt.grid()
plt.locator_params(axis='x', nbins=20)
plt.tight_layout()
# figmanager = plt.get_current_fig_manager()
# figmanager.window.showMaximized()

print("Digits:")
print("For Manhattan the max accuracy was %s at k = %s"%(manhattan_max_acc_digit,manhattan_max_k_digit))
print("For euclidean the max accuracy was %s at k = %s"%(euclidean_max_acc_digit,euclidean_max_k_digit))
print()
print("Faces:")
print("For Manhattan the max accuracy was %s at k = %s"%(manhattan_max_acc_face,manhattan_max_k_face))
print("For euclidean the max accuracy was %s at k = %s"%(euclidean_max_acc_face,euclidean_max_k_face))
print()
print("Total Execution Time: %s seconds" % (time.time() - start_time))

plt.show()