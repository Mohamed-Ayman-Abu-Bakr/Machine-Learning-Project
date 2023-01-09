from sklearn.naive_bayes import *
from sklearn.metrics import accuracy_score
from Data_Load import *
import matplotlib.pyplot as plt
import time
start_time = time.time()

print('///////////////////////////////////////////////////////////////////')
print('//                                                               //')
print('//                          Digits Data                          //')
print('//                                                               //')
print('///////////////////////////////////////////////////////////////////')
print()


DIGIT_TRAINING_SIZE = 5000
DIGIT_TESTING_SIZE = 1000

digits_labels_train, digits_labels_test, digits_train, digits_test = loadDigitsData(DIGIT_TRAINING_SIZE, DIGIT_TESTING_SIZE)

nb1 = GaussianNB()
nb2 = BernoulliNB()
nb3 = ComplementNB()
nb4 = MultinomialNB()
digits_pred1 = nb1.fit(digits_train, digits_labels_train).predict(digits_test)
digits_pred2 = nb2.fit(digits_train, digits_labels_train).predict(digits_test)
digits_pred3 = nb3.fit(digits_train, digits_labels_train).predict(digits_test)
digits_pred4 = nb4.fit(digits_train, digits_labels_train).predict(digits_test)

acc_digit_Gaussian = accuracy_score(digits_labels_test, digits_pred1)*100
acc_digit_Bernoulli = accuracy_score(digits_labels_test, digits_pred2)*100
acc_digit_Complement = accuracy_score(digits_labels_test, digits_pred3)*100
acc_digit_Multinomial = accuracy_score(digits_labels_test, digits_pred4)*100

digits = {"Gaussian":acc_digit_Gaussian,"Bernoulli":acc_digit_Bernoulli,
        "Complement":acc_digit_Complement,"Multinomial":acc_digit_Multinomial}

for key, value in digits.items():
    print(f'Accuracy of', key, ': ', value, '%')

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

nb1 = GaussianNB()
nb2 = BernoulliNB()
nb3 = ComplementNB()
nb4 = MultinomialNB()
faces_pred1 = nb1.fit(faces_train, faces_labels_train).predict(faces_test)
faces_pred2 = nb2.fit(faces_train, faces_labels_train).predict(faces_test)
faces_pred3 = nb3.fit(faces_train, faces_labels_train).predict(faces_test)
faces_pred4 = nb4.fit(faces_train, faces_labels_train).predict(faces_test)

acc_face_Gaussian = accuracy_score(faces_labels_test, faces_pred1)*100
acc_face_Bernoulli = accuracy_score(faces_labels_test, faces_pred2)*100
acc_face_Complement = accuracy_score(faces_labels_test, faces_pred3)*100
acc_face_Multinomial = accuracy_score(faces_labels_test, faces_pred4)*100

faces = {"Gaussian":acc_face_Gaussian,"Bernoulli":acc_face_Bernoulli,
        "Complement":acc_face_Complement,"Multinomial":acc_face_Multinomial}

for key, value in faces.items():
    print(f'Accuracy of', key, ': ', value, '%')

plt.figure('Naive Bayes Classifier',figsize=(12, 5))
plt.subplot(121)
plt.bar(list(digits.keys()), list(digits.values()), width = 0.4)
plt.title('Naive Bayes Classifier for Digits')
plt.xlabel('Distributions')
plt.ylabel('Accuracy (%)')
plt.ylim(50, 80)

plt.subplot(122)
plt.bar(list(faces.keys()), list(faces.values()), width = 0.4)
plt.title('Naive Bayes Classifier for Faces')
plt.xlabel('Distributions')
plt.ylabel('Accuracy (%)')
plt.ylim(49, 52)
# figmanager = plt.get_current_fig_manager()
# figmanager.window.showMaximized()
print("Total Execution Time: %s seconds" % (time.time() - start_time))
plt.show()