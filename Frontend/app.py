from flask import Flask,render_template
import os

TEMPLATE_DIR = os.path.abspath('templates')
STATIC_DIR = os.path.abspath('styles')

app = Flask(__name__,template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)




@app.route('/', methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/behavioural_dash", methods=["GET"])
def behavioural_dash():
    return render_template("escm_dashboard.html")