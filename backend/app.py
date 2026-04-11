import os
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.utils import secure_filename
from sqlalchemy import or_
from ultralytics import YOLO
import time

# --- Get the absolute path for the project directory ---
basedir = os.path.abspath(os.path.dirname(__file__))

# --- AI Model and Calculation Constants ---
MODEL_PATH = os.path.join(basedir, 'model/best.pt') # Use absolute path for model
# These values need to be calibrated once using a stage micrometer.
KNOWN_FOV_AREA_UM2 = 70000.0
ASSUMED_DEPTH_UM = 2.0
# --- End of Constants ---

# 1. App and Database Configuration
# Explicitly tell Flask where to find the templates and static folders
app = Flask(__name__,
            template_folder=os.path.join(basedir, 'templates'),
            static_folder=os.path.join(basedir, 'static'))

app.config['SECRET_KEY'] = '2005'
# Use absolute path for the SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'sih_app.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Configuration for file uploads
UPLOAD_FOLDER = os.path.join(basedir, 'static/uploads') # Use absolute path
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'


# --- DEBUGGING PRINT ---
# This will show in your terminal when the app starts
print("---" * 10)
print(f"Base Directory: {basedir}")
print(f"Template Folder: {app.template_folder}")
print(f"Static Folder: {app.static_folder}")
print(f"Database URI: {app.config['SQLALCHEMY_DATABASE_URI']}")
print(f"Upload Folder: {app.config['UPLOAD_FOLDER']}")
print("---" * 10)
# --- END DEBUGGING ---


@login_manager.user_loader
def load_user(user_id):
    return Doctor.query.get(int(user_id))

# 3. Database Models (No changes here, keeping it for completeness)
class Doctor(db.Model, UserMixin):
    __tablename__ = 'doctors'
    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(50), nullable=False)
    last_name = db.Column(db.String(50), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    phone_number = db.Column(db.String(15), nullable=True)
    doctor_id = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    patients = db.relationship('Patient', backref='doctor', lazy=True)

class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(50), nullable=False)
    last_name = db.Column(db.String(50), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    email = db.Column(db.String(120), nullable=False)
    phone_number = db.Column(db.String(20), nullable=False)
    image_filename = db.Column(db.String(120), nullable=False)
    upload_date = db.Column(db.DateTime, nullable=False, default=db.func.current_timestamp())
    doctor_id = db.Column(db.Integer, db.ForeignKey('doctors.id'), nullable=False)
    rbc_count = db.Column(db.BigInteger, nullable=True)
    wbc_count = db.Column(db.BigInteger, nullable=True)
    platelets_count = db.Column(db.BigInteger, nullable=True)
    result_image_filename = db.Column(db.String(120), nullable=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 4. Application Routes (No changes here)
@app.route('/')
def home():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        hashed_password = bcrypt.generate_password_hash(request.form.get('password')).decode('utf-8')
        new_doctor = Doctor(
            first_name=request.form.get('first_name'),
            last_name=request.form.get('last_name'),
            email=request.form.get('email'),
            phone_number=request.form.get('phone_number'),
            doctor_id=request.form.get('doctor_id'),
            password=hashed_password
        )
        db.session.add(new_doctor)
        db.session.commit()
        flash('Your account has been created! You can now log in.', 'success')
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        doctor = Doctor.query.filter_by(email=request.form.get('email')).first()
        if doctor and bcrypt.check_password_hash(doctor.password, request.form.get('password')):
            login_user(doctor, remember=True)
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Login Unsuccessful. Please check email and password.', 'danger')
    return render_template('login.html')

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/upload_patient', methods=['GET', 'POST'])
@login_required
def upload_patient():
    if request.method == 'POST':
        if 'image' not in request.files or request.files['image'].filename == '':
            flash('No image file selected.', 'danger')
            return redirect(request.url)
        file = request.files['image']
        if file and allowed_file(file.filename):
            timestamp = str(int(time.time()))
            filename = f"{timestamp}_{secure_filename(file.filename)}"
            original_image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(original_image_path)
            model = YOLO(MODEL_PATH)
            results = model(original_image_path)
            predictions = results[0]
            cell_counts = {'RBC': 0, 'WBC': 0, 'Platelets': 0}
            for box in predictions.boxes:
                class_id = int(box.cls[0])
                class_name = predictions.names[class_id]
                if class_name in cell_counts:
                    cell_counts[class_name] += 1
            volume_in_um3 = KNOWN_FOV_AREA_UM2 * ASSUMED_DEPTH_UM
            volume_in_ul = volume_in_um3 / 1e9
            approx_rbc_per_ul = int(cell_counts['RBC'] / volume_in_ul) if volume_in_ul > 0 else 0
            approx_wbc_per_ul = int(cell_counts['WBC'] / volume_in_ul) if volume_in_ul > 0 else 0
            approx_platelets_per_ul = int(cell_counts['Platelets'] / volume_in_ul) if volume_in_ul > 0 else 0
            result_filename = f"result_{filename}"
            result_image_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
            predictions.save(filename=result_image_path)
            new_patient = Patient(
                first_name=request.form.get('first_name'), last_name=request.form.get('last_name'),
                age=request.form.get('age'), gender=request.form.get('gender'),
                email=request.form.get('email'), phone_number=request.form.get('phone_number'),
                image_filename=filename, doctor_id=current_user.id,
                rbc_count=approx_rbc_per_ul, wbc_count=approx_wbc_per_ul,
                platelets_count=approx_platelets_per_ul, result_image_filename=result_filename
            )
            db.session.add(new_patient)
            db.session.commit()
            flash('Patient data uploaded and analysis complete!', 'success')
            return redirect(url_for('view_results', patient_id=new_patient.id))
        else:
            flash('Invalid file type. Please upload a PNG, JPG, or JPEG file.', 'danger')
            return redirect(request.url)
    return render_template('upload_patient.html')

@app.route('/results/<int:patient_id>')
@login_required
def view_results(patient_id):
    patient = Patient.query.get_or_404(patient_id)
    if patient.doctor_id != current_user.id:
        flash('You are not authorized to view this patient\'s data.', 'danger')
        return redirect(url_for('dashboard'))
    normal_ranges = {
        'RBC': (4500000, 5900000), 'WBC': (4500, 11000), 'Platelets': (150000, 450000)
    }
    return render_template('results.html', patient=patient, ranges=normal_ranges)

@app.route('/search_patient', methods=['GET', 'POST'])
@login_required
def search_patient():
    patients = []
    if request.method == 'POST':
        search_query = request.form.get('search_query', '')
        search_term = f"%{search_query}%"
        patients = Patient.query.filter(
            Patient.doctor_id == current_user.id,
            or_(
                Patient.first_name.ilike(search_term),
                Patient.last_name.ilike(search_term),
                Patient.email.ilike(search_term)
            )
        ).all()
    return render_template('search_patient.html', patients=patients)

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('login'))

# 5. Run the Application
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)

