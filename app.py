from flask import Flask, request, render_template, send_file
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import logging
import pandas as pd
import sklearn
import numpy as np
# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
df = pd.read_csv("Placement_Data_Full_Class.csv")
fig, ax = plt.subplots(figsize=(10, 6))
sns.set_style(style="darkgrid")
app = Flask(__name__)
# Set up logging
logging.basicConfig(level=logging.DEBUG)
try:
    with open('combined_model.pkl', 'rb') as f:
        combined_model = pickle.load(f)
        model = combined_model['model']
        preprocessing = combined_model['preprocessing']
except Exception as e:
    logging.error("Error loading the model: %s", str(e))
@app.route('/')
def home():
    return render_template('input.html')
@app.route('/input', methods=['POST'])
def pred():
    try:
        ssc_p = float(request.form.get('ssc_p'))
        hsc_p = float(request.form.get('hsc_p'))
        degree_p = float(request.form.get('degree_p'))
        gender = int(request.form.get('gender'))
        workex = int(request.form.get('workex'))
        etest_p = float(request.form.get('etest_p'))
        specialisation = int(request.form.get('specialisation'))
        mba_p = float(request.form.get('mba_p'))
        # Original input data
        input_data = [[ssc_p, hsc_p, degree_p, gender, workex, etest_p, specialisation, mba_p]]
        # Apply the same preprocessing as in Google Colab Notebook
        scaled_input_data = preprocessing.fit_transform(input_data)
        op = model.predict(scaled_input_data)
        logging.debug("Model output: %s", str(op))
        # Convert the numeric output to "Placed" or "Not Placed"
        result = "Placed" if op[0] == 1 else "Not Placed"
        return render_template('input.html', Output=result)
    except Exception as e:
        logger.error("Error during prediction: %s", str(e))
        return render_template('input.html', Output="Error: " + str(e))
@app.route('/visualize1')
def visualize1():
    fig, ax = plt.subplots(figsize=(10, 6))  # Create new fig and ax objects
    ax.clear()  # Clear previous plot data
    numerical_df = df.select_dtypes(["float64", "int64"])
    values = [(numerical_df['ssc_p'].mean()), (numerical_df['hsc_p'].mean()), (numerical_df['mba_p'].mean()),
              (numerical_df['degree_p'].mean())]
    fig = plt.figure()
    a = fig.add_axes([0, 0, 1, 1])
    names = ['ssc_p', 'hsc_p', 'mba_p', 'degree_p']
    a.set_ylabel('Average percentages')
    a.set_title('Average Percentage')
    bars = a.bar(names, values, width=0.5, color=["#2ca02c"])
    plt.xticks(rotation=45)  # Rotate x-axis tick labels
    a.set_xlabel('Factors')  # Set x-axis label
    a.set_ylabel('Percentage')  # Set y-axis label
    # Add names above the bars
    for bar, name in zip(bars, names):
        yval = bar.get_height()
        xval = bar.get_x() + bar.get_width() / 2
        a.text(xval, yval + 0.1, name, ha='center', va='bottom', color='black', fontweight='bold', fontsize=10)
    # Add percentages below the bars
    for bar in bars:
        yval = bar.get_height()
        xval = bar.get_x() + bar.get_width() / 2
        a.text(xval, yval - 0.05, round(yval, 2), ha='center', va='top', color='black')

    canvas = FigureCanvas(fig)
    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    return send_file(img, mimetype='image/png')


@app.route('/visualize2')
def visualize2():
    fig, ax = plt.subplots(figsize=(10, 6))  # Create new fig and ax objects
    ax.clear()
    total_placed = df[(df["status"] == "Placed")]
    obj1 = total_placed["gender"].value_counts()
    plt.pie(obj1.values, labels=obj1.index, autopct="%2f%%")
    canvas = FigureCanvas(fig)
    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    return send_file(img, mimetype='image/png')


@app.route('/visualize3')
def visualize3():
    fig, ax = plt.subplots(figsize=(10, 6))  # Create new fig and ax objects
    ax.clear()
    plt.bar([1], height=len(df[df["specialisation"] == "Mkt&HR"]))
    plt.bar([0], height=len(df[df["specialisation"] == "Mkt&Fin"]))
    plt.xlabel("specialisation in Mkt&Fin and Mkt&HR")
    plt.ylabel("no.of specialisation")
    plt.xticks(np.arange(2), ('Mkt&Fin', 'Mkt&HR'))
    canvas = FigureCanvas(fig)
    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    return send_file(img, mimetype='image/png')


if __name__ == '__main__':
    app.run(debug=True)
