from sklearn import tree

features = [[340, 2], [240, 6], [440, 1], [380, 1]]
# first feature is highest speed of the car
# second is no of seats

labels = [0, 1, 2, 2]
# 0 = normal car
# 1 = compact car
# 2 = sports car

clf = tree.DecisionTreeClassifier()
clf.fit(features, labels)

print(clf.predict([[350, 4]]))
