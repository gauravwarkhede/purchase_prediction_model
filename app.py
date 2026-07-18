import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# Load the model safely
MODEL_PATH = "naive_mode.pkl"
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
else:
    model = None
    print(f"Warning: {MODEL_PATH} not found. Please ensure it is in the root directory.")

# Cyberpunk UI layout via render_template_string
CYBERPUNK_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NEURAL_NET // PREDICTION_ENGINE</title>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Rajdhani:wght@500;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0a0a12;
            --panel-bg: #121225;
            --neon-cyan: #00f3ff;
            --neon-pink: #ff0055;
            --text-color: #adefd1;
            --grid-color: rgba(0, 243, 255, 0.03);
        }

        body {
            background-color: var(--bg-color);
            background-image: linear-gradient(var(--grid-color) 1px, transparent 1px),
                              linear-gradient(90deg, var(--grid-color) 1px, transparent 1px);
            background-size: 20px 20px;
            color: #ffffff;
            font-family: 'Rajdhani', sans-serif;
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            overflow-x: hidden;
        }

        .container {
            width: 100%;
            max-width: 500px;
            background: var(--panel-bg);
            border: 2px solid var(--neon-cyan);
            box-shadow: 0 0 20px rgba(0, 243, 255, 0.2), inset 0 0 10px rgba(0, 243, 255, 0.1);
            padding: 30px;
            border-radius: 4px;
            position: relative;
            box-sizing: border-box;
        }

        .container::before {
            content: "SYSTEM_STATUS: ONLINE";
            position: absolute;
            top: -12px;
            right: 20px;
            background: var(--bg-color);
            padding: 0 10px;
            font-family: 'Orbitron', sans-serif;
            font-size: 0.75rem;
            color: var(--neon-pink);
            letter-spacing: 2px;
        }

        h1 {
            font-family: 'Orbitron', sans-serif;
            text-align: center;
            color: #fff;
            text-shadow: 0 0 10px var(--neon-cyan);
            font-size: 1.8rem;
            margin-bottom: 30px;
            letter-spacing: 3px;
            text-transform: uppercase;
            border-bottom: 1px dashed var(--neon-cyan);
            padding-bottom: 15px;
        }

        .form-group {
            margin-bottom: 20px;
        }

        label {
            display: block;
            font-size: 1rem;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            margin-bottom: 8px;
            color: var(--neon-cyan);
        }

        select, input {
            width: 100%;
            padding: 12px;
            background: rgba(0, 0, 0, 0.5);
            border: 1px solid rgba(0, 243, 255, 0.3);
            color: #fff;
            font-family: 'Rajdhani', sans-serif;
            font-size: 1.1rem;
            border-radius: 0;
            box-sizing: border-box;
            transition: all 0.3s ease;
        }

        select:focus, input:focus {
            outline: none;
            border-color: var(--neon-pink);
            box-shadow: 0 0 10px rgba(255, 0, 85, 0.5);
        }

        button {
            width: 100%;
            padding: 15px;
            background: transparent;
            border: 2px solid var(--neon-pink);
            color: #fff;
            font-family: 'Orbitron', sans-serif;
            font-size: 1.1rem;
            font-weight: bold;
            letter-spacing: 2px;
            cursor: pointer;
            text-transform: uppercase;
            margin-top: 10px;
            position: relative;
            transition: all 0.2s ease;
        }

        button:hover {
            background: var(--neon-pink);
            box-shadow: 0 0 15px var(--neon-pink);
            text-shadow: 0 0 5px #fff;
        }

        #result-panel {
            margin-top: 25px;
            padding: 15px;
            background: rgba(0, 0, 0, 0.4);
            border-left: 4px solid var(--neon-pink);
            display: none;
            text-align: center;
        }

        .result-title {
            font-size: 0.9rem;
            color: rgba(255,255,255,0.6);
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .result-value {
            font-family: 'Orbitron', sans-serif;
            font-size: 1.6rem;
            color: #fff;
            margin-top: 5px;
            text-shadow: 0 0 8px var(--neon-pink);
        }
    </style>
</head>
<body>

<div class="container">
    <h1>Prediction Engine</h1>
    
    <form id="prediction-form">
        <div class="form-group">
            <label for="gender">Gender</label>
            <select id="gender" name="gender" required>
                <option value="1">Male</option>
                <option value="0">Female</option>
            </select>
        </div>
        
        <div class="form-group">
            <label for="age">Age</label>
            <input type="number" id="age" name="age" min="0" max="120" placeholder="e.g. 28" required>
        </div>
        
        <div class="form-group">
            <label for="salary">Estimated Salary</label>
            <input type="number" id="salary" name="salary" min="0" placeholder="e.g. 50000" required>
        </div>
        
        <button type="submit">Execute Prediction</button>
    </form>

    <div id="result-panel">
        <div class="result-title">// OUTPUT_DATA</div>
        <div class="result-value" id="prediction-output">---</div>
    </div>
</div>

<script>
    document.getElementById('prediction-form').addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const gender = parseFloat(document.getElementById('gender').value);
        const age = parseFloat(document.getElementById('age').value);
        const salary = parseFloat(document.getElementById('salary').value);
        
        const resultPanel = document.getElementById('result-panel');
        const output = document.getElementById('prediction-output');
        
        output.innerText = "PROCESSING...";
        resultPanel.style.display = "block";

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ features: [gender, age, salary] })
            });
            
            const data = await response.json();
            
            if (data.error) {
                output.style.color = "var(--neon-pink)";
                output.innerText = "ERROR: " + data.error;
            } else {
                output.style.color = "var(--neon-cyan)";
                output.style.textShadow = "0 0 8px var(--neon-cyan)";
                output.innerText = "CLASS: " + data.prediction;
            }
        } catch (err) {
            output.style.color = "var(--neon-pink)";
            output.innerText = "CONNECTION FAILED";
        }
    });
</script>

</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(CYBERPUNK_HTML)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model file missing on server"}), 500
        
    try:
        data = request.get_json()
        # Expecting structural array: [Gender, Age, EstimatedSalary]
        features = np.array([data['features']])
        
        # Execute prediction
        prediction = model.predict(features)
        
        return jsonify({"prediction": int(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    # Render binds to the environment variable PORT
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
