from flask import Flask, render_template, request
from src.pipeline.prediction_pipeline import PredictionPipeline, CustomData

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method=="GET":
      return render_template("index.html")
    else:
    # Get form data
      data = CustomData(
          date=request.form.get('date'),
          location=request.form.get('location'),
          temperature=float(request.form.get('temperature')),
          humidity=float(request.form.get('humidity')),
          wind_speed=float(request.form.get('wind_speed')),
          precipitation=float(request.form.get('precipitation')),
          cloud_cover=float(request.form.get('cloud_cover')),
          pressure=float(request.form.get('pressure'))
      )

      final_data = data.get_data_as_dataframe()
      print(final_data)

      predict_pipeline = PredictionPipeline()
      prediction = predict_pipeline.predict(final_data)
      print(prediction)

      prediction_text = "Rain Expected" if prediction[0] == 1 else "No Rain Expected"
     
      return render_template('index.html', prediction=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)