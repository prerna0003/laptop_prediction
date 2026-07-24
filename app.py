import os
import pickle
import numpy as np
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# Load the decision tree model
MODEL_PATH = "decision-tree.pkl"

def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return None

model = load_model()

# Model features in strict order
FEATURE_NAMES = ["Age", "Gender", "Region", "Occupation", "Income"]

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en" data-bs-theme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Decision Tree Predictor - Dark Dashboard</title>
    <!-- Bootstrap 5 CSS Dark Theme -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <!-- FontAwesome Icons -->
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <style>
        body {
            background-color: #0f172a;
            color: #f8fafc;
            min-height: 100vh;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .dashboard-card {
            background-color: #1e293b;
            border: 1px solid #334155;
            border-radius: 16px;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.5);
        }
        .header-box {
            background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%);
            border-radius: 15px 15px 0 0;
            padding: 2rem;
        }
        .form-control, .form-select {
            background-color: #0f172a;
            border: 1px solid #334155;
            color: #f8fafc;
        }
        .form-control:focus, .form-select:focus {
            background-color: #0f172a;
            border-color: #38bdf8;
            color: #f8fafc;
            box-shadow: 0 0 0 0.25rem rgba(56, 189, 248, 0.25);
        }
        .form-label {
            font-weight: 600;
            color: #94a3b8;
            font-size: 0.9rem;
        }
        .btn-predict {
            background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%);
            border: none;
            padding: 12px;
            font-weight: 600;
            border-radius: 10px;
            color: #ffffff;
            transition: all 0.3s ease;
        }
        .btn-predict:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(14, 165, 233, 0.4);
            color: #ffffff;
        }
        .result-card {
            background-color: #0f172a;
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 1.5rem;
        }
    </style>
</head>
<body class="py-5">
    <div class="container">
        <div class="row justify-content-center">
            <div class="col-lg-8">
                <div class="card dashboard-card">
                    <!-- Dashboard Header -->
                    <div class="header-box text-center">
                        <h2 class="fw-bold mb-1 text-white"><i class="fa-solid fa-tree me-2"></i>Decision Tree Classifier</h2>
                        <p class="mb-0 text-white-50">Predict outcomes using customer demographic inputs</p>
                    </div>

                    <div class="card-body p-4 p-md-5">
                        {% if error %}
                            <div class="alert alert-danger d-flex align-items-center" role="alert">
                                <i class="fa-solid fa-circle-exclamation me-2"></i>
                                <div>{{ error }}</div>
                            </div>
                        {% endif %}

                        {% if prediction is not none %}
                            <div class="result-card text-center mb-4">
                                <span class="text-uppercase tracking-wider text-secondary fw-bold small">Prediction Output</span>
                                <h2 class="display-5 fw-bold mt-2 mb-1 {% if prediction == 'yes' %}text-success{% else %}text-danger{% endif %}">
                                    {{ prediction | upper }}
                                </h2>
                                {% if probability is not none %}
                                    <div class="text-muted small">
                                        Model Confidence: <span class="text-info fw-bold">{{ "%.2f"|format(probability * 100) }}%</span>
                                    </div>
                                {% endif %}
                            </div>
                        {% endif %}

                        <form method="POST" action="/">
                            <div class="row g-4">
                                <!-- Age -->
                                <div class="col-md-6">
                                    <label class="form-label">Age</label>
                                    <input type="number" min="18" max="100" name="Age" class="form-control" value="35" required>
                                </div>

                                <!-- Gender -->
                                <div class="col-md-6">
                                    <label class="form-label">Gender</label>
                                    <select name="Gender" class="form-select" required>
                                        <option value="0">Male (0)</option>
                                        <option value="1">Female (1)</option>
                                    </select>
                                </div>

                                <!-- Region -->
                                <div class="col-md-6">
                                    <label class="form-label">Region Code</label>
                                    <input type="number" min="0" max="10" name="Region" class="form-control" value="1" required>
                                </div>

                                <!-- Occupation -->
                                <div class="col-md-6">
                                    <label class="form-label">Occupation Code</label>
                                    <input type="number" min="0" max="10" name="Occupation" class="form-control" value="2" required>
                                </div>

                                <!-- Income -->
                                <div class="col-12">
                                    <label class="form-label">Annual Income ($)</label>
                                    <input type="number" step="any" min="0" name="Income" class="form-control" value="50000" required>
                                </div>
                            </div>

                            <button type="submit" class="btn btn-predict w-100 mt-4">
                                <i class="fa-solid fa-microchip me-2"></i>Generate Prediction
                            </button>
                        </form>
                    </div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    if model is None:
        return render_template_string(HTML_TEMPLATE, prediction=None, probability=None, error="Model file missing on server.")

    prediction = None
    probability = None
    error = None

    if request.method == "POST":
        try:
            features = [float(request.form[feat]) for feat in FEATURE_NAMES]
            input_data = np.array([features])
            
            # Predict outcome class
            pred_class = model.predict(input_data)[0]
            prediction = str(pred_class)

            # Get Confidence Score if available
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(input_data)[0]
                class_idx = list(model.classes_).index(pred_class)
                probability = float(probs[class_idx])

        except Exception as e:
            error = f"Error during prediction: {str(e)}"

    return render_template_string(HTML_TEMPLATE, prediction=prediction, probability=probability, error=error)


@app.route("/predict", methods=["POST"])
def api_predict():
    if model is None:
        return jsonify({"error": "Model file missing"}), 500

    try:
        data = request.get_json(force=True)
        features = [float(data[feat]) for feat in FEATURE_NAMES]
        input_data = np.array([features])
        
        pred_class = model.predict(input_data)[0]
        response = {"prediction": str(pred_class)}
        
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_data)[0]
            class_idx = list(model.classes_).index(pred_class)
            response["probability"] = float(probs[class_idx])

        return jsonify(response)
    except KeyError as e:
        return jsonify({"error": f"Missing input parameter: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
