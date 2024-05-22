from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import subprocess

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'  # TODO: Change
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to process images with your model


def process_image(clothing_path, selfie_path):
    output_path = "/home/lizcar/MagicClothing/output_img"
    # Command to invoke the model inference script
    command = f"CUDA_VISIBLE_DEVICES=2 python inference.py --cloth_path {clothing_path} --person_path {selfie_path} --model_path /home/lizcar/MagicClothing/MagicClothing/magic_clothing_768_vitonhd_joint.safetensors --output_path {output_path}"

    try:
        # Execute the command
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        # Extract the path of the processed image from the output
        # print("std out: ", result.stdout, result.stderr)
        # output_lines = result.stdout.split('\n')
        # result_image_path = output_lines[0].strip()
        return output_path
    except Exception as e:
        print(f"Error processing image: {e}")
        return None


@app.route('/upload', methods=['POST'])
def upload():
    # Check if the post request has the file part
    if 'clothing' not in request.files or 'selfie' not in request.files:
        return jsonify({"error": "Missing files in request"}), 400

    clothing = request.files['clothing']
    selfie = request.files['selfie']

    # Check if the file is empty
    if clothing.filename == '' or selfie.filename == '':
        return jsonify({"error": "One or more selected files are empty"}), 400

    # Save the file to the UPLOAD_FOLDER
    clothing_path = os.path.join(app.config['UPLOAD_FOLDER'], clothing.filename)
    selfie_path = os.path.join(app.config['UPLOAD_FOLDER'], selfie.filename)
    clothing.save(clothing_path)
    selfie.save(selfie_path)

    # Process image with your model
    result_image_path = process_image(clothing_path, selfie_path)

    if result_image_path is None:
        return jsonify({"error": "Failed to process images"}), 500

    # Return the path of the processed image
    return jsonify({"result_image_path": result_image_path}), 200


@app.route('/result/<filename>')
def get_result(filename):
    # Send the result image file back to the client
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3030)
