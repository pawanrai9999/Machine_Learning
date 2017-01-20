from sklearn import tree

features = [[140, 0], [130, 0], [150, 1], [170, 1]]
# features for classifier 
# weight is in gram
# 0 = smooth texture
# 1 = rought texture

label = [0, 0, 1, 1]
# 0 = Apple
# 1 = Orange

clf = tree.DecisionTreeClassifier()
clf.fit(features, label)

print(clf.predict([[180, 1]]))
