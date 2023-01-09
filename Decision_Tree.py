from sklearn import tree
from sklearn.metrics import accuracy_score
from Data_Load import *
import matplotlib.pyplot as plt
from statistics import mean
plt.rcParams['figure.constrained_layout.use'] = True
# import time
# start_time = time.time()

SHOW_PLOTS = False


def getResults(data,**params):
    dt = tree.DecisionTreeClassifier(**params)
    model = dt.fit(data["Training Data"], data["Training Labels"])
    pred = model.predict(data["Testing Data"])
    accuracy = accuracy_score(data["Testing Labels"], pred)*100

    return model, accuracy

criterions = {
    "Entropy":"entropy",
    "Gini": "gini",
    "Log Loss": "log_loss"
    }

print('///////////////////////////////////////////////////////////////////')
print('//                                                               //')
print('//                          Digits Data                          //')
print('//                                                               //')
print('///////////////////////////////////////////////////////////////////')
print()

DIGIT_TRAINING_SIZE = 5000
DIGIT_TESTING_SIZE = 1000
digits_labels_train, digits_labels_test, digits_train, digits_test = loadDigitsData(DIGIT_TRAINING_SIZE, DIGIT_TESTING_SIZE)

data = {
    "Training Data":digits_train,
    "Training Labels":digits_labels_train,
    "Testing Data":digits_test,
    "Testing Labels":digits_labels_test
    }

best_params = dict()
print("Testing Criterion:")
criterion_res = dict()
for parameter, value in criterions.items():
    accuracies=[]
    for _ in range(10):
        model, accuracy = getResults(data=data,criterion=value)
        accuracies.append(accuracy)
    average = mean(accuracies)
    criterion_res[parameter]=average
    print(f'Accuracy of {parameter} Criterion : {average} %')

best_params["criterion"]={"param_value":criterions[max(criterion_res,key=criterion_res.get)],"accuracy":max(criterion_res.values())}

plt.figure('Decision Tree Statistics for Digits',figsize=(19.2, 10.8))
plt.subplot(231)
plt.bar(list(criterion_res.keys()),list(criterion_res.values()),width = 0.4)
plt.title('Accuracy Based on Criterion')
plt.xlabel('Criterion Type')
plt.ylabel('Accuracy (%)')
plt.ylim(70, 76)

print("\nTesting Max Depth:")

depth_res = dict()

for depth in range(1,21):
    model, accuracy = getResults(data,max_depth = depth)
    depth_res[depth]=accuracy
    print(f'Accuracy at depth {depth} : {accuracy} %')

best_params["max_depth"]={"param_value":max(depth_res,key=depth_res.get),"accuracy":max(depth_res.values())}

plt.subplot(232)
plt.plot(list(depth_res.keys()),list(depth_res.values()))
plt.title('Accuracy Based on Depth of tree')
plt.xlabel('Depth')
plt.ylabel('Accuracy (%)')
plt.locator_params(axis='x', nbins=20)
plt.grid()

print("\nTesting minimum number of samples before splitting:")

min_samples_split_res = dict()
for min_samples_split in range(2,31):
    model, accuracy = getResults(data,min_samples_split = min_samples_split)
    min_samples_split_res[min_samples_split]=accuracy
    print(f'Accuracy when minimum samples for split = {min_samples_split} : {accuracy} %')
    
best_params["min_samples_split"]={"param_value":max(min_samples_split_res,key=min_samples_split_res.get),"accuracy":max(min_samples_split_res.values())}

plt.subplot(233)
plt.plot(list(min_samples_split_res.keys()),list(min_samples_split_res.values()))
plt.title('Accuracy Based on minimum samples for split')
plt.xlabel('Minimum number of samples for split')
plt.ylabel('Accuracy (%)')
plt.locator_params(axis='x', nbins=20)
plt.grid()

print("\nTesting minimum samples for a leaf node:")

min_samples_leaf_res = dict()
for min_samples_leaf in range(1,31):
    model, accuracy = getResults(data,min_samples_leaf = min_samples_leaf)
    min_samples_leaf_res[min_samples_leaf]=accuracy
    print(f'Accuracy when minimum samples for leaf = {min_samples_leaf} : {accuracy} %')
    
best_params["min_samples_leaf"]={"param_value":max(min_samples_leaf_res,key=min_samples_leaf_res.get),"accuracy":max(min_samples_leaf_res.values())}

plt.subplot(234)
plt.plot(list(min_samples_leaf_res.keys()),list(min_samples_leaf_res.values()))
plt.title('Accuracy Based on minimum samples for leaf')
plt.xlabel('Minimum number of samples for leaf')
plt.ylabel('Accuracy (%)')
plt.locator_params(axis='x', nbins=20)
plt.grid()

print("\nTesting maximum number of leaf nodes:")


max_leaf_nodes_res = dict()
for max_leaf_nodes in range(2,1000,50):
    model, accuracy = getResults(data,max_leaf_nodes = max_leaf_nodes)
    max_leaf_nodes_res[max_leaf_nodes]=accuracy
    print(f'Accuracy when maximum number of leaf nodes = {max_leaf_nodes} : {accuracy} %')

best_params["max_leaf_nodes"]={"param_value":max(max_leaf_nodes_res,key=max_leaf_nodes_res.get),"accuracy":max(max_leaf_nodes_res.values())}

plt.subplot(235)
plt.plot(list(max_leaf_nodes_res.keys()),list(max_leaf_nodes_res.values()))
plt.title('Accuracy Based on maximum number of leaf nodes')
plt.xlabel('Maximum number of leaf nodes')
plt.ylabel('Accuracy (%)')
plt.locator_params(axis='x', nbins=20)
plt.grid()

print("\nDigits Results:")

best_res = dict()
for parameter, res in best_params.items():
    value = res["param_value"]
    accuracy = res["accuracy"]
    print(f'Best accuracy for parameter {parameter} was at value {value} with accuracy : {accuracy} %')
    best_res[f"{parameter}={value}"]=accuracy

temp_params = dict()
for key, value in best_params.items():
    temp_params[key]=value["param_value"]
model, accuracy = getResults(data,**temp_params)
best_res["All Best Parameters"]=accuracy
print(f'Best accuracy with all best parameters : {accuracy} %')


plt.subplot(236)
plt.bar(list(best_res.keys()),list(best_res.values()),width=0.4)
plt.title('Best accuracy for each parameter')
plt.xlabel('Parameter')
plt.ylabel('Accuracy (%)')
plt.ylim(73,78)
plt.xticks(rotation=45, ha='right')
plt.savefig('./figures/decision_tree_digits_statistics.png')
if SHOW_PLOTS:
    plt.show()

plt.figure('Decision Tree for Digits',figsize=(19.2, 10.8))
tree.plot_tree(model, filled=True, fontsize=5)
plt.savefig('./figures/decision_tree_digits.png')
if SHOW_PLOTS:
    plt.show()


print('//////////////////////////////////////////////////////////////////')
print('//                                                              //')
print('//                          Faces Data                          //')
print('//                                                              //')
print('//////////////////////////////////////////////////////////////////')
print()

FACE_TRAINING_SIZE = 451
FACE_TESTING_SIZE = 150

faces_labels_train, faces_labels_test, faces_train, faces_test = loadFacesData(FACE_TRAINING_SIZE, FACE_TESTING_SIZE)


data = {
    "Training Data":faces_train,
    "Training Labels":faces_labels_train,
    "Testing Data":faces_test,
    "Testing Labels":faces_labels_test
    }


best_params = dict()

print("\nTesting Criterion:")


criterion_res = dict()
for parameter, value in criterions.items():
    accuracies=[]
    for _ in range(10):
        model, accuracy = getResults(data=data,criterion=value)
        accuracies.append(accuracy)
    average = mean(accuracies)
    criterion_res[parameter]=average
    print(f'Accuracy of {parameter} Criterion : {average} %')

best_params["criterion"]={"param_value":criterions[max(criterion_res,key=criterion_res.get)],"accuracy":max(criterion_res.values())}

plt.figure('Decision Tree Statistics for Faces',figsize=(19.2, 10.8))
plt.subplot(231)
plt.bar(list(criterion_res.keys()),list(criterion_res.values()),width = 0.4)
plt.title('Accuracy Based on Criterion')
plt.xlabel('Criterion Type')
plt.ylabel('Accuracy (%)')
plt.ylim(45, 55)


print("\nTesting Max Depth:")

depth_res = dict()

for depth in range(1,21):
    model, accuracy = getResults(data,max_depth = depth)
    depth_res[depth]=accuracy
    print(f'Accuracy at depth {depth} : {accuracy} %')

best_params["max_depth"]={"param_value":max(depth_res,key=depth_res.get),"accuracy":max(depth_res.values())}

plt.subplot(232)
plt.plot(list(depth_res.keys()),list(depth_res.values()))
plt.title('Accuracy Based on Depth of tree')
plt.xlabel('Depth')
plt.ylabel('Accuracy (%)')
plt.locator_params(axis='x', nbins=20)
plt.grid()

print("\nTesting minimum number of samples before splitting:")


min_samples_split_res = dict()
for min_samples_split in range(2,31):
    model, accuracy = getResults(data,min_samples_split = min_samples_split)
    min_samples_split_res[min_samples_split]=accuracy
    print(f'Accuracy when minimum samples for split = {min_samples_split} : {accuracy} %')
    
best_params["min_samples_split"]={"param_value":max(min_samples_split_res,key=min_samples_split_res.get),"accuracy":max(min_samples_split_res.values())}

plt.subplot(233)
plt.plot(list(min_samples_split_res.keys()),list(min_samples_split_res.values()))
plt.title('Accuracy Based on minimum samples for split')
plt.xlabel('Minimum number of samples for split')
plt.ylabel('Accuracy (%)')
plt.locator_params(axis='x', nbins=20)
plt.grid()

print("\nTesting minimum samples for a leaf node:")


min_samples_leaf_res = dict()
for min_samples_leaf in range(1,31):
    model, accuracy = getResults(data,min_samples_leaf = min_samples_leaf)
    min_samples_leaf_res[min_samples_leaf]=accuracy
    print(f'Accuracy when minimum samples for leaf = {min_samples_leaf} : {accuracy} %')
    
best_params["min_samples_leaf"]={"param_value":max(min_samples_leaf_res,key=min_samples_leaf_res.get),"accuracy":max(min_samples_leaf_res.values())}

plt.subplot(234)
plt.plot(list(min_samples_leaf_res.keys()),list(min_samples_leaf_res.values()))
plt.title('Accuracy Based on minimum samples for leaf')
plt.xlabel('Minimum number of samples for leaf')
plt.ylabel('Accuracy (%)')
plt.locator_params(axis='x', nbins=20)
plt.grid()

print("\nTesting maximum number of leaf nodes:")

max_leaf_nodes_res = dict()
for max_leaf_nodes in range(2,1000,50):
    model, accuracy = getResults(data,max_leaf_nodes = max_leaf_nodes)
    max_leaf_nodes_res[max_leaf_nodes]=accuracy
    print(f'Accuracy when maximum number of leaf nodes = {max_leaf_nodes} : {accuracy} %')

best_params["max_leaf_nodes"]={"param_value":max(max_leaf_nodes_res,key=max_leaf_nodes_res.get),"accuracy":max(max_leaf_nodes_res.values())}

plt.subplot(235)
plt.plot(list(max_leaf_nodes_res.keys()),list(max_leaf_nodes_res.values()))
plt.title('Accuracy Based on maximum number of leaf nodes')
plt.xlabel('Maximum number of leaf nodes')
plt.ylabel('Accuracy (%)')
plt.locator_params(axis='x', nbins=20)
plt.grid()

print("\nFacess Results:")

best_res = dict()
for parameter, res in best_params.items():
    value = res["param_value"]
    accuracy = res["accuracy"]
    print(f'Best accuracy for parameter {parameter} was at value {value} with accuracy : {accuracy} %')
    best_res[f"{parameter}={value}"]=accuracy

temp_params = dict()
for key, value in best_params.items():
    temp_params[key]=value["param_value"]
model, accuracy = getResults(data,**temp_params)
best_res["All Best Parameters"]=accuracy
print(f'Best accuracy with all best parameters : {accuracy} %')


plt.subplot(236)
plt.bar(list(best_res.keys()),list(best_res.values()),width=0.4)
plt.title('Best accuracy for each parameter')
plt.xlabel('Parameter')
plt.ylabel('Accuracy (%)')
plt.ylim(45,65)
plt.xticks(rotation=45, ha='right')
plt.savefig('./figures/decision_tree_faces_statistics.png')
if SHOW_PLOTS:
    plt.show()

plt.figure('Decision Tree for Faces',figsize=(19.2, 10.8))
tree.plot_tree(model, filled=True, fontsize=5)
plt.savefig('./figures/decision_tree_faces.png')
if SHOW_PLOTS:
    plt.show()