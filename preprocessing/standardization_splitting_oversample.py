# STANDARDIZATION
data = features.values[:,:-1]

scalar=StandardScaler()
data = scalar.fit_transform(data)
data = DataFrame(data)
data['character'] = features['character']

# SPLITTING (maintaining proportions between characters' dialogue lines)
X_train,X_test,Y_train,Y_test = train_test_split(data.drop(['character'], axis=1),data['character'],test_size=0.30, shuffle=True, stratify = data['character'])
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

# OVERSAMPLE
# visualizing distirbution of dialogue lines between characters in the train set
counter = Counter(Y_train)
for k,v in counter.items():
	per = v / len(Y_train) * 100
	print(k, v, "%.2f" % per+'%')
# plot the distribution
pyplot.bar(counter.keys(), counter.values())
pyplot.show()

# data augmentation for the less represented classes
oversample = SMOTE()
X_train_res, Y_train_res = oversample.fit_resample(X_train, Y_train)
print(X_train_res.shape, Y_train_res.shape, X_test.shape, Y_test.shape)

# visualizing new distirbution of dialogue lines between characters in the train set
counter = Counter(Y_train_res)
for k,v in counter.items():
	per = v / len(Y_train_res) * 100
	print(k, v, "%.2f" % per+'%')
# plot the distribution
pyplot.bar(counter.keys(), counter.values())
pyplot.show()
