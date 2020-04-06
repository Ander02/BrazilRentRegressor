import pandas as pd
import os
import matplotlib as plt
import tensorflow.keras as keras
from flask import Flask, escape, request
from Model import create_model, train

app = Flask(__name__)

model = None     
database_path = "./database/houses_to_rent.csv"
save_path = "./model.h5"

@app.route("/model/new", methods=["GET", "POST"])
def generateNewModel():
    model = create_model()
    train(model=model, database_path=database_path, save_path=save_path)
    model = keras.models.load_model(save_path)


@app.route("/model/load", methods=["GET", "POST"])
def loadModel():
    model = keras.models.load_model("teste.h5")


@app.route("/model/load", methods=["GET"])
def testModel():
    pass


if __name__ == "__main__":
    app.run()