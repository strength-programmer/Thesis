from flask import Flask, render_template, Response, jsonify, request, send_from_directory, session, redirect, url_for, flash
import sys
import os
from datetime import datetime
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))
import livefeed
from functools import wraps
from models import db, Employee
from config import Config

app = Flask(__name__)
app.config.from_object(Config)
db.init_app(app)

# Create database tables
with app.app_context():
    db.create_all()

# Start the background video thread when the app starts
livefeed.start_background_video_thread()

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route("/login", methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Replace this with your actual user authentication logic
        if username == "username" and password == "password":  # Example credentials
            session['logged_in'] = True
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password')
    
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

@app.route("/")
@login_required
def home():
    return render_template("home.html")

@app.route("/about")
@login_required
def about():
    return render_template("about.html")

@app.route("/livefeed")
@login_required
def livefeed_page():
    return render_template("livefeed.html")

@app.route("/recordings")
@login_required
def recordings():
    return render_template("recordings.html")

@app.route("/employees")
@login_required
def employees():
    return render_template("employees.html")

@app.route("/account")
@login_required
def account():
    return render_template("account.html")

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            frame = livefeed.get_latest_frame()
            if frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                import time
                time.sleep(0.05)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/activity_status')
def activity_status():
    return jsonify({"active": livefeed.get_activity_recognition_state()})

@app.route('/toggle_activity', methods=['POST'])
def toggle_activity():
    new_state = livefeed.toggle_activity_recognition()
    return jsonify({"active": new_state})

@app.route('/toggle_recording', methods=['POST'])
def toggle_recording():
    if not hasattr(livefeed.video_manager, 'writer') or livefeed.video_manager.writer is None:
        livefeed.video_manager.setup_recording()
        return jsonify({"recording": True})
    else:
        livefeed.video_manager.stop_recording()
        return jsonify({"recording": False})

@app.route('/recording_status')
def recording_status():
    is_recording = hasattr(livefeed.video_manager, 'writer') and livefeed.video_manager.writer is not None
    return jsonify({"recording": is_recording})

@app.route('/download_recording/<filename>')
def download_recording(filename):
    return send_from_directory('recordings', filename, as_attachment=True)

@app.route('/list_recordings')
def list_recordings():
    recordings_dir = os.path.join(os.path.dirname(__file__), 'recordings')
    if not os.path.exists(recordings_dir):
        os.makedirs(recordings_dir)
    
    files = [f for f in os.listdir(recordings_dir) if f.endswith('.mp4')]
    return jsonify({"files": files})

@app.route('/recordings/<filename>')
def serve_recording(filename):
    recordings_dir = os.path.join(os.path.dirname(__file__), 'recordings')
    return send_from_directory(recordings_dir, filename)

@app.route('/delete_recording/<filename>', methods=['DELETE'])
def delete_recording(filename):
    try:
        recordings_dir = os.path.join(os.path.dirname(__file__), 'recordings')
        file_path = os.path.join(recordings_dir, filename)
        
        if os.path.exists(file_path):
            os.remove(file_path)
            return jsonify({"success": True})
        else:
            return jsonify({"success": False, "error": "File not found"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

# Employee CRUD routes
@app.route('/api/employees', methods=['GET'])
@login_required
def get_employees():
    employees = Employee.query.all()
    return jsonify([employee.to_dict() for employee in employees])

@app.route('/api/employees/<int:id>', methods=['GET'])
@login_required
def get_employee(id):
    employee = Employee.query.get_or_404(id)
    return jsonify(employee.to_dict())

@app.route('/api/employees', methods=['POST'])
@login_required
def create_employee():
    data = request.json
    
    # Convert hire_date string to date object
    hire_date = datetime.strptime(data['hireDate'], '%Y-%m-%d').date()
    
    employee = Employee(
        employee_id=data['employeeId'],
        full_name=data['fullName'],
        photo_url=data.get('photoUrl'),
        role=data['role'],
        hire_date=hire_date,
        email=data['email'],
        phone=data.get('phone'),
        status=data['status'],
        department=data.get('department')
    )
    
    try:
        db.session.add(employee)
        db.session.commit()
        return jsonify(employee.to_dict()), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 400

@app.route('/api/employees/<int:id>', methods=['PUT'])
@login_required
def update_employee(id):
    employee = Employee.query.get_or_404(id)
    data = request.json
    
    # Convert hire_date string to date object if provided
    if 'hireDate' in data:
        data['hire_date'] = datetime.strptime(data['hireDate'], '%Y-%m-%d').date()
    
    # Update fields
    for key, value in data.items():
        if hasattr(employee, key):
            setattr(employee, key, value)
    
    try:
        db.session.commit()
        return jsonify(employee.to_dict())
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 400

@app.route('/api/employees/<int:id>', methods=['DELETE'])
@login_required
def delete_employee(id):
    employee = Employee.query.get_or_404(id)
    
    try:
        db.session.delete(employee)
        db.session.commit()
        return '', 204
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5050)
