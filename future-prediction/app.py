from flask import Flask, request, jsonify
from flask_cors import CORS
from predict import predict

app = Flask(__name__)
CORS(app)  # allows your website to call this API from a different domain


@app.route("/predict", methods=["GET"])
def get_prediction():
    region   = request.args.get("region")
    date_str = request.args.get("date")

    if not region or not date_str:
        return jsonify({"error": "Missing required params: region, date"}), 400

    try:
        result = predict(
            region=region,
            date_str=date_str,
            rgb_channels=(0, 1, 2),
            stretch_rgb=True
        )
        return jsonify({
            "region":    result["region"],
            "date":      result["date"],
            "source":    result["source"],
            "image_b64": result["image_b64"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
