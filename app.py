from flask import Flask, request, jsonify
from master_et_predictor import predict_ET

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    et, scenario = predict_ET(data)

    return jsonify({
        "Scenario": scenario,
        "ET": et
    })


if __name__ == "__main__":
    app.run(debug=True)
