from flask import Flask, request, render_template
from gradio_client import Client, handle_file
import tempfile
import os
import shutil  # To copy files

app = Flask(__name__)
client = Client("yisol/IDM-VTON")

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if 'background_image' not in request.files or 'garment_image' not in request.files:
            return "Please upload both background and garment images."

        background_image = request.files['background_image']
        garment_image = request.files['garment_image']

        # Save the uploaded files to temporary paths
        bg_temp = tempfile.NamedTemporaryFile(delete=False)
        garm_temp = tempfile.NamedTemporaryFile(delete=False)
        background_image.save(bg_temp.name)
        garment_image.save(garm_temp.name)

        # Make the API request to the virtual try-on endpoint
        result = client.predict(
            dict={"background": handle_file(bg_temp.name), "layers": [], "composite": None},
            garm_img=handle_file(garm_temp.name),
            garment_des="",
            is_checked=True,       # Default value
            is_checked_crop=False, # Default value
            denoise_steps=30,      # Default value
            seed=42,               # Default value
            api_name="/tryon"
        )

        # Close the temporary files
        bg_temp.close()
        garm_temp.close()

        # Remove the temporary files
        os.remove(bg_temp.name)
        os.remove(garm_temp.name)

        # Assuming result contains paths for the images
        if isinstance(result, tuple) and len(result) >= 2:
            output_image_path = result[0]
            masked_image_path = result[1]

            # Copy the output images to the static folder to serve them
            static_output_path = "static/output_image.png"
            static_masked_path = "static/masked_image.png"

            shutil.copy(output_image_path, static_output_path)
            shutil.copy(masked_image_path, static_masked_path)

            return render_template("index.html", output_image=static_output_path, masked_image=static_masked_path)
        else:
            return f"Unexpected result format from the API: {result}"

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
