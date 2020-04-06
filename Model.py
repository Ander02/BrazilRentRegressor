import numpy as np
import tensorflow.keras as keras
import pandas as pd
import matplotlib.pyplot as plt

def create_model():
    model = keras.Sequential()
    model.add(keras.layers.Dense(units=15, activation="relu"))
    model.add(keras.layers.Dense(units=15, activation="relu"))
    model.add(keras.layers.Dense(units=15, activation="relu"))
    model.add(keras.layers.Dense(units=1, activation="sigmoid"))
    return model

def getCallbacks():

    earlyStopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=300)
    terminateOnNan = keras.callbacks.TerminateOnNaN()    

    return [earlyStopping, terminateOnNan]

def train(model, database_path, save_path, batch_size=10000, epochs=5000, optimizer="rmsprop", metrics=["MeanAbsoluteError", "RootMeanSquaredError"]):
    
    #Process data
    data_frame = pd.read_csv(database_path)

    ##Change furniture to bool
    furniture_dict = {"furnished":1, "not furnished":0 }
    data_frame["furniture_bool"] = data_frame["furniture"].map(furniture_dict)

    ##Change accept animal to bool
    animal_dict = {"acept":1, "not acept":0 }
    data_frame["animal_bool"] = data_frame["animal"].map(animal_dict)

    ##Drop Columns
    data_frame = data_frame.drop(columns=["city", "floor", "furniture", "animal"])

    ##Remove outliers
    column_to_verify = data_frame["rent amount (R$)"]
    outliers = column_to_verify.between(column_to_verify.quantile(.05), column_to_verify.quantile(.95))

    data_frame = data_frame[outliers]

    ##Shuffle
    np.random.shuffle(data_frame.to_numpy())

    #Separe Data        
    labels = data_frame["rent amount (R$)"].to_numpy()
    data = data_frame.drop(columns=["rent amount (R$)"]).to_numpy()

    #Compile Model
    model.compile(loss="MSLE", optimizer=optimizer, metrics=metrics)
    
    #Fit Model
    history = model.fit(x=data[:7000], y=labels[:7000],
                        batch_size=batch_size,
                        epochs=epochs,
                        shuffle=True,
                        callbacks=getCallbacks(),
                        validation_data=[data[-3000:], labels[-3000:]],
                        verbose=1)

    #Save Model
    model.save(save_path)
    #Analyse
    predicts = model.predict(data)
    # plt.scatter(range(len(predicts)), predicts)
    # plt.scatter(range(len(labels)), labels)
    # plt.show()

    plt.plot(predicts.flatten() - labels)
    plt.show()
    #Plot or return the training progress

    plt.scatter(history.epoch, history.history["RootMeanSquaredError"])
    plt.scatter(history.epoch, history.history["val_RootMeanSquaredError"])
    plt.show()
    
    plt.scatter(history.epoch, history.history["MeanAbsoluteError"])
    plt.scatter(history.epoch, history.history["val_MeanAbsoluteError"])
    plt.show()

    plt.scatter(history.epoch, history.history["loss"])
    plt.scatter(history.epoch, history.history["val_loss"])
    plt.show()
    pass
   