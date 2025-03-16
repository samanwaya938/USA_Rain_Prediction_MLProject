from flask import Flask, render_template, request, route

app = Flask(__name__)

@app.route("/")
def home():
  return render_template("index.html")

def predict(data):
  pass

if __name__ == "__main__":
  app.run(debug=True)

