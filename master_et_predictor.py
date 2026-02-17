import numpy as np
import joblib
from tensorflow.keras.models import load_model

def predict_ET(inputs):

    keys = inputs.keys()

    if all(k in keys for k in ['Tmax','Tmin','RHmax','RHmin','n','u2']):
        S="S1"; cols=['Tmax','Tmin','RHmax','RHmin','n','u2']

    elif all(k in keys for k in ['Tmax','Tmin','n','u2']):
        S="S2"; cols=['Tmax','Tmin','n','u2']

    elif all(k in keys for k in ['Tmax','Tmin','u2']):
        S="S3"; cols=['Tmax','Tmin','u2']

    elif all(k in keys for k in ['Tmax','Tmin','RHmax','RHmin','n']):
        S="S4"; cols=['Tmax','Tmin','RHmax','RHmin','n']

    elif all(k in keys for k in ['Tmax','Tmin','RHmax','RHmin','u2']):
        S="S5"; cols=['Tmax','Tmin','RHmax','RHmin','u2']

    elif all(k in keys for k in ['Tmax','Tmin','RHmax','RHmin']):
        S="S6"; cols=['Tmax','Tmin','RHmax','RHmin']

    elif all(k in keys for k in ['Tmax','Tmin','n']):
        S="S7"; cols=['Tmax','Tmin','n']

    else:
        return "Insufficient inputs"

    model = load_model(f"ET_APP_MODELS/{S}/ANN_Ludhiana_{S}.keras")
    scalerX = joblib.load(f"ET_APP_MODELS/{S}/scalerX_{S}.save")
    scalery = joblib.load(f"ET_APP_MODELS/{S}/scalery_{S}.save")

    X = np.array([[inputs[c] for c in cols]])
    Xs = scalerX.transform(X)

    ET = model.predict(Xs)
    ET = scalery.inverse_transform(ET)

    return float(ET[0][0]), S


if __name__ == "__main__":

    sample = {
        "Tmax":30,
        "Tmin":18,
        "n":7,
        "u2":2
    }

    et,scenario = predict_ET(sample)
    print("Scenario used:",scenario)
    print("Predicted ET:",et)
