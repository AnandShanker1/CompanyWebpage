from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
import shortuuid
import math
from sympy import symbols
import numpy as np
from scipy.optimize import fsolve


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['IMAGE_FOLDER'] = 'static/images/graphs/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['IMAGE_FOLDER'], exist_ok=True)


ALLOWED_EXTENSIONS = {'csv'}

# Function to create prediction images
def failure_model_w_CI(sku, warranty_DC):
    data = warranty_DC.copy()
    newdata = warranty_DC[warranty_DC['parent_Parent_Sku'] == sku]

    sortedData = newdata['days_diff'].dropna().sort_values(ascending = True)
    sortedData = pd.DataFrame(sortedData)
    sortedData = sortedData[sortedData['days_diff'] <= 2160]
    timeSum = float(sortedData.sum().iloc[0])
    beta = symbols("beta")
    n = len(sortedData)
    bVal = sortedData.iloc[-1]

    bVal = float(bVal.iloc[0]) if isinstance(bVal, (np.ndarray, pd.Series)) else bVal
    beta = float(beta.iloc[0]) if isinstance(beta, (np.ndarray, pd.Series)) else beta

    def equation_beta(beta):
        return timeSum + (n/beta) - (n * bVal)/(1 - math.exp(-beta * bVal))

    x1 = fsolve(equation_beta, 0.001)
    beta = x1[0]
    alpha = math.log((n*(beta))/((math.e)**(beta*bVal)-1))

    
    def f(t):
        return (np.exp(alpha + beta * t) - np.exp(alpha)) / beta 

    def CI(t, beta, alpha):
        return (np.exp(alpha + beta * t) - np.exp(alpha)) / beta 

    
    failure_counts = sortedData['days_diff'].value_counts().sort_index()
    cumulative_failures = failure_counts.cumsum()
    
    counts = sortedData['days_diff'].value_counts().sort_index()
    pdf = counts / counts.sum()

    x = np.linspace(0, 2160, 2160)
    y = f(x)
    x1 = cumulative_failures.index
    y1 = cumulative_failures.values  

    fisher_info = n / (beta**2) - (n * bVal**2 * np.exp(-beta * bVal)) / ((1 - np.exp(-beta * bVal))**2)
    se_beta = np.sqrt(1 / fisher_info)
    z_score = 1.96
    beta_hat_ci_lower = beta - z_score * se_beta
    beta_hat_ci_upper = beta + z_score * se_beta
    alpha_hat_ci_lower = np.log((n * beta_hat_ci_lower) / (np.exp(beta_hat_ci_lower * bVal) - 1))
    alpha_hat_ci_upper = np.log((n * beta_hat_ci_upper) / (np.exp(beta_hat_ci_upper * bVal) - 1))

    beta_hat_ci_lower = float(beta_hat_ci_lower)
    beta_hat_ci_upper = float(beta_hat_ci_upper)
    alpha_hat_ci_lower = float(alpha_hat_ci_lower)
    alpha_hat_ci_upper = float(alpha_hat_ci_upper)

    y_predicted_ci_lower = CI(np.array(x), beta_hat_ci_lower, alpha_hat_ci_lower)
    y_predicted_ci_upper = CI(np.array(x), beta_hat_ci_upper, alpha_hat_ci_upper)

    plt.figure(figsize=(12, 7)) 
    
    plt.fill_between(
        x,
        y_predicted_ci_lower,
        y_predicted_ci_upper,
        color="#e31a1c",
        alpha=0.3,
        label="95% CI",
    )

    # Plot the first curve (predicted function)
    plt.plot(x, y, color='red', linestyle='-', label="Predicted Curve")
    
    # Plot the second curve (cumulative failures)
    plt.plot(x1, y1, color='blue', linestyle='-', label="Actual Cumulative Failures")
    
    # Labels and title
    plt.xlabel("Time (Days)")
    plt.ylabel("Cumulative Failures")
    plt.title(f"Predicted vs. Actual Cumulative Failures for {sku}")
    
    # Add grid and legend
    plt.grid(True)
    plt.legend()

    # Save the plot to the static/images/ folder
    file_name = f"{shortuuid.uuid()}.png"
    plt.savefig(f"{app.config['IMAGE_FOLDER']}{file_name}")
    
    return file_name    

# Function to check if the uploaded file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route for rendering the main index page
@app.route('/')
def index():
    return render_template('index.html')

# Route to clear all files from the static/images/ directory
@app.route('/clear')
def clear(): 
    for file in os.listdir(app.config['IMAGE_FOLDER']):
        file_path = os.path.join(app.config['IMAGE_FOLDER'], file)
        if os.path.isfile(file_path): 
            os.remove(file_path)
    return jsonify(success=True)

# Route to handle file uploads
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part', 400  # Return error if no file is provided
    
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400  # Return error if no file is selected
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)  # Save uploaded file

        image_list = []

        warranty_DC = pd.read_csv(filepath)  # Read uploaded CSV file

        parent_sku_list = []
        for sku in warranty_DC['parent_Parent_Sku']:
            if sku not in parent_sku_list:
                parent_sku_list.append(sku)

        # Generate images based on SKU data and store them in image_list
        for sku in parent_sku_list:
            sku_img = failure_model_w_CI(sku, warranty_DC)
            image_list.append(sku_img)

        os.remove(filepath)  # Remove the uploaded file after processing
        return render_template("graphs.html", image_list=image_list)
    
    return 'Invalid file type', 400  # Return error if file type is not allowed

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
