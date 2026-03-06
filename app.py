from flask import Flask, request, render_template


import pickle



file1 = open('bodyfatmodel.pkl', 'rb')
rf = pickle.load(file1)
file1.close()


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def predict():

    if request.method == 'POST':

        density = float(request.form['density'])
        abdomen = float(request.form['abdomen'])
        chest = float(request.form['chest'])
        weight = float(request.form['weight'])
        hip = float(request.form['hip'])

        import pandas as pd

        input_features = pd.DataFrame(
            [[density, abdomen, chest, weight, hip]],
            columns=["Density","Abdomen","Chest","Weight","Hip"]
        )

        prediction = rf.predict(input_features)[0].round(2)

        string = 'Percentage of Body Fat Estimated is : ' + str(prediction) + '%'

        return render_template('show.html', string=string)

    return render_template('home.html')


if __name__ == "__main__":
    app.run(debug=True)
