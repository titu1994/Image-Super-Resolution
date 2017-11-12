from keras.models import load_model
from flask import Flask, request, render_template, flash, redirect, url_for
from werkzeug.utils import secure_filename

import models
import os
import tensorflow as tf

upload_folder = 'data/'
if not os.path.exists(upload_folder):
   os.makedirs(upload_folder)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

model = load_model('keras_models/RNSR_model.h5')
with tf.device('/cpu:0'):
    m = models.ResNetSR(2)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = upload_folder
app.secret_key = 'WTF_I_already*Installed^%Open%&$CV'

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def root():

    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file',
                            filename=filename))

    return render_template('index.html', title='Image Super Resolution')

@app.route('/uploaded_file/<string:filename>')
def uploaded_file(filename):
    path = upload_folder + filename

    try:
        m.upscale(path, save_intermediate=False, mode="fast")

        ext = filename.rsplit('.', 1)[1]
        path = upload_folder + filename.rsplit('.', 1)[0] + "_scaled(2x)." + ext

        return redirect(url_for('image', filename=path))
    except:
        flash("Image is too large !")
        return redirect('/')


@app.route('/image/<filename>', methods=['POST'])
def image(filename):
    return render_template('disp.html', image=filename)


if __name__ == "__main__":
    app.run(port=8888)