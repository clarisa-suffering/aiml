from flask import Flask, render_template, request
from model import calculate_tdee, recommend_meals

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        age = int(request.form["age"])
        weight = float(request.form["weight"])
        height = float(request.form["height"])
        gender = request.form["gender"]
        activity = request.form["activity"]
        goal = request.form["goal"]

        tdee = calculate_tdee(gender, weight, height, age, activity, goal)
        meals = recommend_meals(tdee)

        return render_template("index.html", tdee=tdee, meals=meals)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
