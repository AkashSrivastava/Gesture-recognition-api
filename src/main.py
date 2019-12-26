from flask import Flask, jsonify, request
import sys
from assignment_2 import models
import assignment_2

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def hello():
    content = request.get_json()
    prediction = models.preprocessing(content)
    return jsonify({"1": prediction[0].lower(),
                    "2": prediction[1].lower(),
                    "3": prediction[2].lower(),
                    "4": prediction[3].lower()})



if __name__ == "__main__":
    app.run()
