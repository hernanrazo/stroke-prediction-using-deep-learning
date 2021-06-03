import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import ADASYN
import keras.utils as utils
from keras import layers, models
from keras.layers import Conv2D, Flatten, Dense, Dropout, BatchNormalization
from keras.layers.core import Activation
from keras.optimizers import Adam
from keras.regularizers import l2

# stop warnings from printing
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def main():

    # get data
    try:
        data = pd.read_csv('~/pythonProjects/stroke-pred/healthcare-dataset-stroke-data.csv')
    except:
        print('ERROR: File path to dataset is not correct. Make sure to adjust it to fit your system before running.')
        exit()

    # clean the dataset in preparation for training
    # remove the useless id column
    # fill null values with the mean
    data.drop('id', axis=1, inplace=True)
    data = data.fillna(data['bmi'].mean())

    # turn gender to numeric
    data['gender'] = np.where((data.gender == 'Male'), '0', data['gender'])
    data['gender'] = np.where((data.gender == 'Female'), '1', data['gender'])
    data['gender'] = np.where((data.gender == 'Other'), '2', data['gender'])
    data['gender'] = data['gender'].astype('int')

    # turn marriage to numeric
    data['ever_married'] = np.where((data.ever_married == 'No'), '0', data['ever_married'])
    data['ever_married'] = np.where((data.ever_married == 'Yes'), '1', data['ever_married'])
    data['ever_married'] = data["ever_married"].astype('int')

    # turn work types to numeric
    data['work_type'] = np.where((data.work_type == 'Private'), '0', data['work_type'])
    data['work_type'] = np.where((data.work_type == 'Self-employed'), '1', data['work_type'])
    data['work_type'] = np.where((data.work_type == 'Govt_job'), '2', data['work_type'])
    data['work_type'] = np.where((data.work_type == 'children'), '3', data['work_type'])
    data['work_type'] = np.where((data.work_type == 'Never_worked'), '4', data['work_type'])
    data['work_type'] = data['work_type'].astype('int')

    # turn residence to numeric
    data['Residence_type'] = np.where((data.Residence_type == 'Urban'), '0', data['Residence_type'])
    data['Residence_type'] = np.where((data.Residence_type == 'Rural'), '1', data['Residence_type'])
    data['Residence_type'] = data['Residence_type'].astype('int')

    # turn smoking to numeric
    data['smoking_status'] = np.where((data.smoking_status == 'formerly smoked'), '0', data['smoking_status'])
    data['smoking_status'] = np.where((data.smoking_status == 'never smoked'), '1', data['smoking_status'])
    data['smoking_status'] = np.where((data.smoking_status == 'smokes'), '2', data['smoking_status'])
    data['smoking_status'] = np.where((data.smoking_status == 'Unknown'), '3', data['smoking_status'])
    data['smoking_status'] = data['smoking_status'].astype('int')

    # turn certain age ranges into bins of values instead of several individual values
    data.loc[data['age'] <= 18, 'age'] = 0  # children
    data.loc[(data['age'] > 18) & (data['age'] <= 35), 'age'] = 1 # adults
    data.loc[(data['age'] > 35) & (data['age'] <= 65), 'age'] = 2 # older adults
    data.loc[(data['age'] > 65) , 'age'] = 3 # elderly
    data['age'] = data['age'].astype('int')

    # Turn certain glucose ranges into bins according to data given
    # by the American Diabetes Association. Levels include normal,
    # pre-diabetes, and diabetes
    # https://www.diabetes.org/a1c/diagnosis
    data.loc[data['avg_glucose_level'] <= 100, 'avg_glucose_level'] = 0 # normal
    data.loc[(data['avg_glucose_level'] > 100) & (data['avg_glucose_level'] <= 125), 'avg_glucose_level'] = 1 # pre-diabetes
    data.loc[(data['avg_glucose_level'] > 125), 'avg_glucose_level'] = 2 # diabetes
    data['avg_glucose_level'] = data['avg_glucose_level'].astype('int')

    # turn certain BMI ranges into bins according to the Center for
    # Disease Control. Levels include underweight, normal, overweight,
    # and obese
    # https://www.cdc.gov/healthyweight/assessing/bmi/adult_bmi/index.html
    data.loc[data['bmi'] <= 18.5, 'bmi'] = 0 # underweight
    data.loc[(data['bmi'] > 18.5) & (data['bmi'] <= 24.9), 'bmi'] = 1 # normal
    data.loc[(data['bmi'] > 24.9) & (data['bmi'] <= 29.9), 'bmi'] = 2 # overweight
    data.loc[(data['bmi'] > 29.9) , 'bmi'] = 3 # obese

    # split independant and dependant variables
    x = data.drop('stroke', axis=1)
    y = data['stroke']

    # deal with sample imbalancing
    ada = ADASYN()
    x, y = ada.fit_sample(x, y)

    # split training, testing, validation
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    x_val = x_train[:500]
    y_val = y_train[:500]

    # recreation of the nerual network stated in the research paper (Cheon et al.)
    # Input layer: 11 neurons, relu activation, L2 regularization, dropout=0.2
    model = models.Sequential()
    model.add(Dense(11, input_dim=10, activation='relu', kernel_regularizer=l2(0.0001)))
    model.add(Dropout(0.2))

    # Hidden layer: 22 neurons, relu activation, dropout=0.2
    model.add(Dense(22, activation='relu', kernel_regularizer=l2(0.0001)))
    model.add(Dropout(0.2))

    # Hidden layer: 10 neurons, relu activation, dropout=0.2, batch norm
    model.add(Dense(10, activation='relu', kernel_regularizer=l2(0.0001)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    # Hidden layer: 10 neurons, relu activation, dropout=0.2, batch norm
    model.add(Dense(10, activation='relu', kernel_regularizer=l2(0.0001)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    # Hidden layer: 10 neurons, relu activation, dropout=0.2, batch norm
    model.add(Dense(10, activation='relu', kernel_regularizer=l2(0.0001)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    # Output layer: 1 neuron (regression output)
    model.add(Dense(1, activation='relu'))

    print(model.summary())

    # train the model
    # binary crossentropy, adam optimizer, learning rate=0.001, L2 regulatization
    model.compile(optimizer=Adam(lr=0.001),
                  loss='binary_crossentropy',
                  metrics=['binary_accuracy'])
    history = model.fit(x_train,
                        y_train,
                        epochs=50,
                        batch_size=5,
                        validation_data=(x_val, y_val))

    print('Performance: ', model.evaluate(x_test, y_test))

    # Plot loss and accuracy
    epochs = range(1, len(history.history['loss']) +1)
    plt.plot(epochs, history.history['loss'], color='blue', marker='o', label='Training loss')
    plt.plot(epochs, history.history['val_loss'], color='green', marker='o', label='Validation loss')
    plt.title('Training and Validation Loss: Model #1')
    plt.xlabel('# of Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss1.png')
    plt.clf()

    plt.plot(epochs, history.history['binary_accuracy'], color='blue', marker='o', label='Training accuracy')
    plt.plot(epochs, history.history['val_binary_accuracy'], color='green', marker='o', label='Validation accuracy')
    plt.title('Training and Validation Accuracy: Model #1')
    plt.xlabel('# of Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('acc1.png')
    plt.clf()

    model.save('model1.h5')

    del model, history, epochs

    # Perform a new implementation that has a custom neural network
    # Input layer: 64 neurons, relu activation, initializer as uniform
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(10,), kernel_initializer='uniform'))

    # Hidden layer: 32 neurons, relu activation
    model.add(layers.Dense(32, activation='relu'))

    # Hidden layer: 16 neurons, relu activation
    model.add(layers.Dense(16, activation='relu'))

    # Hidden layer: 10 neurons, relu activation
    model.add(layers.Dense(10, activation='relu'))

    # Output layer: 1 neuron, sigmoid activation (regression output)
    model.add(layers.Dense(1, activation='sigmoid'))
    print(model.summary())

    model.compile(optimizer=Adam(0.001),
                  loss='binary_crossentropy',
                  metrics=['binary_accuracy'])

    history = model.fit(x_train,
                        y_train,
                        epochs=30,
                        batch_size=5,
                        validation_data=(x_val, y_val))

    print('Performance:')
    print(model.evaluate(x_test, y_test))

    # Plot loss and accuracy
    epochs = range(1, len(history.history['loss']) +1)
    plt.plot(epochs, history.history['loss'], color='blue', marker='o', label='Training loss')
    plt.plot(epochs, history.history['val_loss'], color='green', marker='o', label='Validation loss')
    plt.title('Training and Validation Loss: Model #2')
    plt.xlabel('# of Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss2.png')
    plt.clf()

    plt.plot(epochs, history.history['binary_accuracy'], color='blue', marker='o', label='Training accuracy')
    plt.plot(epochs, history.history['val_binary_accuracy'], color='green', marker='o', label='Validation accuracy')
    plt.title('Training and Validation Accuracy: Model #2')
    plt.xlabel('# of Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('acc2.png')
    plt.clf()

    model.save('model2.h5')
if __name__ == '__main__':
    main()
