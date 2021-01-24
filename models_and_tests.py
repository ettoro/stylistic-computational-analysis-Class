# LOGISTIC REGRESSION
logmodel = LogisticRegression(multi_class='multinomial', solver='newton-cg', max_iter=100000)
logmodel.fit(X_train_res,Y_train_res)
predictions = logmodel.predict(X_test)
print(classification_report(Y_test,predictions))

# RANDOM FOREST
RF_class = RandomForestClassifier(random_state=1)
RF_class.fit(X_train_res, Y_train_res)
y_pred_rf = RF_class.predict(X_test)
print(classification_report(Y_test, y_pred_rf))

# KNN
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train_res, Y_train_res)
pred = knn.predict(X_test)
print(classification_report(Y_test, pred))

# SVC
clf = svm.SVC(kernel='linear')
clf.fit(X_train_res, Y_train_res)
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(Y_test, y_pred))

# KERAS NEURAL NETWORK
X = data.iloc[:, :-1]
y = data.iloc[:,-1]
enc = OneHotEncoder()
Y = enc.fit_transform(y[:, np.newaxis]).toarray() # encoding characters' names

# splitting and oversampling again with the encoding
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, stratify = Y, random_state=4)
n_features = X.shape[1]
n_classes = Y.shape[1]
print(n_features, n_classes)

oversample = SMOTE()
X_train_res, Y_train_res = oversample.fit_resample(X_train, Y_train)
print(X_train_res.shape, Y_train_res.shape, X_test.shape, Y_test.shape)

# model creation
def create_custom_model(input_dim, output_dim, nodes, n=1, name='model'):
    def create_model():
        # Create model
        model = Sequential(name=name)
        for i in range(n):
            model.add(Dense(nodes, input_dim=input_dim, activation='relu'))
        model.add(Dense(output_dim, activation='softmax'))

        # Compile model
        model.compile(loss='categorical_crossentropy', 
                      optimizer='adam', 
                      metrics=['accuracy'])
        return model
    return create_model

models = [create_custom_model(n_features, n_classes, 8, i, 'model_{}'.format(i)) 
          for i in range(1, 4)]

for create_model in models:
    create_model().summary()
    

history_dict = {}
cb = TensorBoard()
for create_model in models:
    model = create_model()
    print('Model name:', model.name)
    history_callback = model.fit(X_train_res, Y_train_res,
                                 batch_size=5,
                                 epochs=10,
                                 verbose=0,
                                 validation_data=(X_test, Y_test),
                                 callbacks=[cb])
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    
    history_dict[model.name] = [history_callback, model]

# plotting epochs and validation loss
fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 6))

for model_name in history_dict:
    val_accurady = history_dict[model_name][0].history['val_accuracy']
    val_loss = history_dict[model_name][0].history['val_loss']
    ax1.plot(val_accurady, label=model_name)
    ax2.plot(val_loss, label=model_name)
    
ax1.set_ylabel('validation accuracy')
ax2.set_ylabel('validation loss')
ax2.set_xlabel('epochs')
ax1.legend()
ax2.legend();

# score
create_model = create_custom_model(n_features, n_classes, 8, 3)

estimator = KerasClassifier(build_fn=create_model, epochs=100, batch_size=5, verbose=0)
scores = cross_val_score(estimator, X_train, Y_train, cv=10)
print("Accuracy : {:0.2f} (+/- {:0.2f})".format(scores.mean(), scores.std()))

# roc curve
plt.figure(figsize=(10, 10))
plt.plot([0, 1], [0, 1], 'k--')

for model_name in history_dict:
    model = history_dict[model_name][1]
    
    Y_pred = model.predict(X_test)
    fpr, tpr, threshold = roc_curve(Y_test.ravel(), Y_pred.ravel())
    
    plt.plot(fpr, tpr, label='{}, AUC = {:.3f}'.format(model_name, auc(fpr, tpr)))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend()

