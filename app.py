import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

b = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
     12, 13, 14, 15, 16, 17, 18, 19, 20, 21]

a = ['Apel', 'Pisang', 'Kacang Hitam', 'Buncis', 'Kelapa', 'Kopi', 'Kapas', 'Anggur', 'Rami', 'Kacang Merah', 'Kacang Lentil',
     'Jagung', 'Mangga', 'Kacang Matki', 'Kacang Hijau', 'Muskmelon', 'Jeruk', 'Pepaya', 'Kacang Gude', 'Delima', 'Padi', 'Semangka']

a = pd.DataFrame(a, columns=['label'])
b = pd.DataFrame(b, columns=['encoded'])
classes = pd.concat([a, b], axis=1).sort_values('encoded').set_index('label')


def predict_proba(n, p, k, temp, humid, pH, rain):
    data = [[n, p, k, temp, humid, pH, rain]]
    pred = model.predict_proba(data)
    pred = pd.DataFrame(data=np.round(pred.T*100, 2),
                        index=classes.index, columns=['predicted_values'])
    high = pred.predicted_values.nlargest(5)
    return high


@app.route('/')
def welcome():
    return render_template('index.html')

@app.route('/list-tanaman')
def crop():
    return render_template('crop.html')

@app.route('/', methods=['POST'])
def predict():
    n = request.form.get('n')
    p = request.form.get('p')
    k = request.form.get('k')
    temp = request.form.get('Temperature')
    humid = request.form.get('Humidity')
    pH = request.form.get('PH')
    rain = request.form.get('rain_fall')
    data = [[n, p, k, temp, humid, pH, rain]]
    pred2 = model.predict(data)
    pred1 = predict_proba(n, p, k, temp, humid, pH, rain)
    fig, axes = plt.subplots()

    plt.title('5 Rekomendasi Tanaman untuk Budidaya',fontdict={'fontsize': 20, 'fontweight': 'medium'})
    axes.pie(x=pred1, autopct='%1.1f%%', labels=pred1.index, explode=(0.1, 0, 0, 0, 0), shadow=True, startangle=90)

    img = BytesIO()
    plt.savefig(img, format='png', transparent=True)
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')

    for i in range(0, len(classes)):
        if(classes.encoded[i]==pred2):
            output=classes.index[i].upper()
    return render_template('index.html',plot_url=plot_url, prediction_text=format(output))

    # return render_template('index.html', plot_url=plot_url)


if __name__ == "__main__":
    app.run(debug=True)
