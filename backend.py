from flask import Flask, render_template, request, redirect, url_for, session
from flask_cors import CORS
import numpy as np
import os
import uuid
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow as tf

app = Flask(__name__)
app.secret_key = 'clave_secreta_para_login'
CORS(app)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

USUARIO = "admin"
PASSWORD = "1234"

# Cargar modelo TFLite
interpreter = None
try:
    interpreter = tf.lite.Interpreter(model_path='modelo/brain_tumor_cnn.tflite')
    interpreter.allocate_tensors()
    print("Modelo TFLite cargado correctamente")
except Exception as e:
    print(f"Error al cargar modelo: {e}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def procesar_imagen(filepath):
    img = Image.open(filepath).convert('RGB')
    img = img.resize((128, 128))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Añadir dimensión batch
    return img_array

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/validar', methods=['POST'])
def validar():
    usuario = request.form['usuario']
    password = request.form['password']
    
    if usuario == USUARIO and password == PASSWORD:
        session['usuario'] = usuario
        return redirect(url_for('principal'))
    return redirect(url_for('login'))

@app.route('/principal')
def principal():
    if 'usuario' not in session:
        return redirect(url_for('login'))
    return render_template('principal.html', usuario=session['usuario'])

@app.route('/clasificar')
def clasificar_view():
    if 'usuario' not in session:
        return redirect(url_for('login'))
    return render_template('clasificador.html')

@app.route('/analizar', methods=['POST'])
def analizar():
    if 'usuario' not in session:
        return redirect(url_for('login'))
    
    if 'imagen' not in request.files:
        return redirect(url_for('clasificar_view'))
    
    file = request.files['imagen']
    
    if file.filename == '':
        return redirect(url_for('clasificar_view'))
    
    if file and allowed_file(file.filename):
        filename = secure_filename(str(uuid.uuid4()) + '_' + file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            img_array = procesar_imagen(filepath)
            
            # Obtener detalles de entrada/salida del modelo
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            # Verificar forma esperada
            expected_shape = input_details[0]['shape']
            print(f"Forma esperada: {expected_shape}")
            print(f"Forma enviada: {img_array.shape}")
            
            interpreter.set_tensor(input_details[0]['index'], img_array)
            interpreter.invoke()
            prediccion = interpreter.get_tensor(output_details[0]['index'])
            
            idx = np.argmax(prediccion[0])
            confianza = float(prediccion[0][idx])
            
            clases = ['glioma', 'meningioma', 'notumor', 'pituitary']
            clase = clases[idx]
            
            if clase == 'notumor':
                resultado = f"TEJIDO NORMAL - Sin tumor detectado\nConfianza: {confianza*100:.2f}%"
                clase_css = "normal"
            else:
                resultado = f"{clase.upper()} DETECTADO - Tumor detectado\nConfianza: {confianza*100:.2f}%"
                clase_css = "tumor"
            
            return render_template('resultado.html', resultado=resultado, clase_css=clase_css)
            
        except Exception as e:
            return render_template('resultado.html', resultado=f"Error al procesar: {str(e)}", clase_css="error")
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
    
    return redirect(url_for('clasificar_view'))

@app.route('/salir')
def salir():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))