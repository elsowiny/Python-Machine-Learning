from flask import Flask, render_template, request

app = Flask(__name__)

# add urls for navigation as a dict
urls = {
    "knn": "/knn",
}

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/lr')
def linear_regression():
    return render_template('LR.html')


if __name__ == "__main__":
    app.run(debug=True)