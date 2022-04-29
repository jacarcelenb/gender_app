from flask import Flask, render_template, request
import modelo

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


def predecir():
    if request.method == "POST":
        pelo = request.form['pelo']
        anchofrente = request.form['anchofrente']
        alturafrente = request.form['alturafrente']
        narizancha= request.form['narizancha']
        narizalta = request.form['narizalta']
        labios = request.form['labios']
        distancia = request.form['distancia']
       

        entradas = [float(pelo), float(anchofrente), float(alturafrente),
                    float(narizancha), float(narizalta), float(labios),
                    float(distancia)]

        prediccion = modelo.predecirGenero(entradas)

    return render_template("test.html", genero=prediccion)

if __name__ == '__main__':
    app.run()
