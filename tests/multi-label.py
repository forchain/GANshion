# use mlp for prediction on multi-label classification
from numpy import asarray
from sklearn.datasets import make_multilabel_classification
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# get the dataset
def get_dataset():
    X, y = make_multilabel_classification(n_samples=1000, n_features=10, n_classes=3, n_labels=2, random_state=1)
    return X, y


# get the model
def get_model(n_inputs, n_outputs):
    model = Sequential()
    model.add(Dense(20, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(n_outputs, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model


if __name__ == '__main__':

    # load dataset
    X, y = get_dataset()
    n_inputs, n_outputs = X.shape[1], y.shape[1]
    # get model
    model = get_model(n_inputs, n_outputs)
    # fit the model on all data
    model.fit(X, y, verbose=0, epochs=100)
    # make a prediction for new data
    row = [3, 3, 6, 7, 8, 2, 11, 11, 1, 3]
    newX = asarray([row])
    yhat = model.predict(newX)
    print('Predicted: %s' % yhat[0])