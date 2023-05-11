# TODO: add data provisioning code here


def get_prosaic_context(verseRefString):
    # TODO: add rest of function body here
    return verseRefString[-1::-1]


#######################################
# Accept post methods to API endpoint #
#######################################

from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route("/api", methods=["POST"])
def api():
    data = request.get_json()  # Get the JSON data sent in the POST
    verseRefString = data.get(
        "verseRefString"
    )  # Get the verseRefString value from the JSON data
    result = get_prosaic_context(verseRefString)  # Execute your function
    return jsonify(result)  # Return the result as JSON


if __name__ == "__main__":
    app.run(debug=True)
