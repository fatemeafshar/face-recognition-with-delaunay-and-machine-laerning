from DelaunayTriangulation import delaunay
from sklearn.neighbors import KNeighborsClassifier
from Delauni.read import read_data

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

import numpy as np
num_classes = 7

def shuffle(triangles_features, labels):
    import random
    c = list(zip(triangles_features, labels))
    random.shuffle(c)
    triangles_features, labels = zip(*c)
    return triangles_features, labels


def k_nearest_neighbor(triangles_features, labels):
    neigh = KNeighborsClassifier(n_neighbors=num_classes)

    # neigh.fit(triangles_features, labels)
    return neigh
    # KNeighborsClassifier(...)

    # print(neigh.predict([]))
    #
    #
    # print(neigh.predict_proba([[0.9]]))

def naive_bayse(triangles_features, labels):

    # X, y = load_iris(return_X_y=True)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
    gnb = GaussianNB()
    # gnb.fit(triangles_features, labels)

    # y_pred = gnb.fit(triangles_features, labels).predict(X_test)
    # print("Number of mislabeled points out of a total %d points : %d"
    #       % (X_test.shape[0], (y_test != y_pred).sum()))
    return gnb

def svm(triangles_features, labels):
    from sklearn import svm
    # X = [[0, 0, 0 ], [1, 1,1]]
    # y = [0, 1]
    clf = svm.SVC()
    # clf.fit(triangles_features, labels)
    return clf
def decision_tree(triangles_features, labels):
    from sklearn import tree
    # X = [[0, 0], [1, 1]]
    # Y = [0, 1]
    clf = tree.DecisionTreeClassifier()
    # clf = clf.fit(triangles_features, labels)
    return clf
def quadratic_classifier(triangles_features, labels):
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    import numpy as np
    # X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    # y = np.array([1, 1, 1, 2, 2, 2])
    clf = QuadraticDiscriminantAnalysis()
    # clf.fit(triangles_features, labels)
    # print(clf.predict([[-0.8, -1]]))
    return clf
def random_forest(triangles_features, labels):
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    import numpy as np
    # X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    # y = np.array([1, 1, 1, 2, 2, 2])
    clf = QuadraticDiscriminantAnalysis()
    # clf.fit(triangles_features, labels)
    # print(clf.predict([[-0.8, -1]]))
    return clf



# #prepare data
# delaunay_object =delaunay()
# path = "/home/fateme/Documents/archive/images/images/train/angry/412.jpg"
# f = delaunay_object.delaunay_tirangulation(path)
# print(f)
def prepare_data(path, chunk):
    # path = "/home/fateme/Documents/archive/images/train/"

    r = read_data()
    pathes, classes_label, labels = r.read(path, chunk)
    triangles_features = []
    labels = []
    delaunay_object = delaunay()
    for i in range(len(pathes)):
        one_image_features = delaunay_object.delaunay_tirangulation(pathes[i])
        if one_image_features != []:
            triangles_features.append(one_image_features)
            labels.append(int(i/chunk))
    # print(triangles_features)
    print("labels", labels)


    most_triangles_image = 0
    for t in triangles_features:
        if len(t) > most_triangles_image:
            most_triangles_image = len(t)
    most_triangles_image = 960
    for i in range(len(triangles_features)):
        for j in range(most_triangles_image - len(triangles_features[i])):
            triangles_features[i].append(0)#[(0,0), (0,0), (0,0), (0,0)])

    for t in triangles_features:
        print(len(t),t)
    triangles_features, labels = shuffle(triangles_features, labels)
    return triangles_features, labels

def visualize(model, triangles_features, labels, triangle_feature_test, labels_test):
    model.fit(triangles_features, labels)
    wrong = 0
    for i in range(len(triangle_feature_test)):
        print("test", model.predict([triangle_feature_test[i]]), str(labels_test[i]))
        if model.predict([triangle_feature_test[i]]) != labels_test[i]:
            wrong += 1
    print("accuracy : ", 1-wrong/len(labels_test))
# prepare train data
triangles_features, labels = prepare_data("/home/fateme/Documents/archive/images/train/", 420)


# triangles_features = np.array(triangles_features)
# nsamples, nx, ny = triangles_features.shape
# triangles_features = triangles_features.reshape((nsamples,nx*ny))
# knn_model = k_nearest_neighbor(triangles_features, classes)
# naive_bayse_model = naive_bayse()

#prepate test data
triangle_feature_test, labels_test = prepare_data("/home/fateme/Documents/archive/images/validation/", 5)
model = k_nearest_neighbor(triangles_features, labels)

model = naive_bayse(triangles_features, labels)

model = svm(triangles_features, labels)

model = decision_tree(triangles_features, labels)


model = random_forest(triangles_features, labels)

model = quadratic_classifier(triangles_features, labels)
visualize(model, triangles_features, labels, triangle_feature_test, labels_test)



#train models

# from matplotlib import image
# from matplotlib import pyplot as plt
#
#
# # to read the image stored in the working directory
# img = image.imread(path)
# # to draw a line from (200,300) to (500,100)
# # x = [200, 500]
# # y = [300, 100]
# # plt.plot(x, y, color="white", linewidth=3)
# d = []
# for coordinate in f[0]:
#     d.append([coordinate[0],coordinate[1]])
#     plt.plot(coordinate[0],coordinate[1], marker="o", markersize=5, markeredgecolor="red", markerfacecolor="green")
#
#
# # points = np.array(d)
# # print(tri.simplices)
# # plt.triplot(points[:,0], points[:,1], tri.simplices)
# plt.imshow(img)
# plt.show()