from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

MODEL_DIR = 'model'
MODEL_NAME = 'obesity._knn_pipeline.pkl'
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)
