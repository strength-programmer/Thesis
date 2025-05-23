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
                # Short sleep to prevent CPU spinning when no frames available
                import time
                time.sleep(0.01)  # Reduced from 0.05 to make it more responsive
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
    data = request.get_json()
    recording_type = data.get('type', 'original') if data else 'original'
    
    current_status = livefeed.get_recording_status()
    
    if recording_type == 'original':
        if not current_status['original']:
            livefeed.start_recording('original')
            return jsonify({"recording": True, "type": "original"})
        else:
            livefeed.stop_recording('original')
            return jsonify({"recording": False, "type": "original"})
    else:  # segmented
        if not current_status['segmented']:
            livefeed.start_recording('segmented')
            return jsonify({"recording": True, "type": "segmented"})
        else:
            livefeed.stop_recording('segmented')
            return jsonify({"recording": False, "type": "segmented"})

@app.route('/toggle_dual_recording', methods=['POST'])
def toggle_dual_recording():
    """Toggle both original and segmented recording simultaneously"""
    current_status = livefeed.get_recording_status()
    
    # If either recording is active, stop both
    if current_status['original'] or current_status['segmented']:
        livefeed.stop_dual_recording()
        return jsonify({"recording": False, "message": "Both recordings stopped"})
    else:
        # Start both recordings
        success = livefeed.start_dual_recording()
        return jsonify({"recording": success, "message": "Both recordings started" if success else "Failed to start recordings"})

@app.route('/recording_status')
def recording_status():
    status = livefeed.get_recording_status()
    return jsonify(status)

@app.route('/download_recording/<filename>')
def download_recording(filename):
    return send_from_directory('recordings', filename, as_attachment=True)

# @app.route('/list_recordings')
# def list_recordings():
#     recordings_dir = os.path.join(os.path.dirname(__file__), 'recordings')
#     if not os.path.exists(recordings_dir):
#         os.makedirs(recordings_dir)
    
#     files = [f for f in os.listdir(recordings_dir) if f.endswith('.mp4')]
#     return jsonify({"files": files})

# @app.route('/recordings/<filename>')
# def serve_recording(filename):
#     recordings_dir = os.path.join(os.path.dirname(__file__), 'recordings')
#     return send_from_directory(recordings_dir, filename)

@app.route('/list_recordings')
def list_recordings():
    recordings_dir = os.path.join(os.path.dirname(__file__), 'recordings')
    segmented_dir = os.path.join(recordings_dir, 'segmented_recordings')
    
    if not os.path.exists(recordings_dir):
        os.makedirs(recordings_dir)
    if not os.path.exists(segmented_dir):
        os.makedirs(segmented_dir)
    
    files = []
    processing_files = []
    
    # Get original recordings (excluding raw segmented files)
    if os.path.exists(recordings_dir):
        original_files = [f for f in os.listdir(recordings_dir) 
                         if f.endswith(('.mp4', '.avi')) 
                         and os.path.isfile(os.path.join(recordings_dir, f))
                         and not f.startswith('raw_segmented_recording_')]  # Exclude raw files
        files.extend(original_files)
        
        # Check for raw segmented files that are being processed
        raw_files = [f for f in os.listdir(recordings_dir) 
                    if f.startswith('raw_segmented_recording_') and f.endswith(('.mp4', '.avi'))]
        processing_files.extend(raw_files)
    
    # Get segmented recordings
    if os.path.exists(segmented_dir):
        segmented_files = [f for f in os.listdir(segmented_dir) 
                          if f.endswith(('.mp4', '.avi'))]
        # Add path prefix to distinguish segmented recordings
        files.extend([f"segmented_recordings/{f}" for f in segmented_files])
    
    return jsonify({
        "files": files,
        "processing": processing_files  # Files currently being processed
    })

@app.route('/recordings/<path:filename>')
def serve_recording(filename):
    import mimetypes
    recordings_dir = os.path.join(os.path.dirname(__file__), 'recordings')
    
    # Handle both direct files and segmented_recordings subdirectory
    if filename.startswith('segmented_recordings/'):
        # Remove the prefix and look in segmented_recordings subdirectory
        actual_filename = filename.replace('segmented_recordings/', '')
        file_path = os.path.join(recordings_dir, 'segmented_recordings', actual_filename)
        serve_dir = os.path.join(recordings_dir, 'segmented_recordings')
        serve_filename = actual_filename
    else:
        # Look in main recordings directory
        file_path = os.path.join(recordings_dir, filename)
        serve_dir = recordings_dir
        serve_filename = filename
    
    # Check if file exists
    if not os.path.exists(file_path):
        return "File not found", 404
    
    # Get file info
    file_size = os.path.getsize(file_path)
    
    # Determine MIME type
    mime_type, _ = mimetypes.guess_type(serve_filename)
    if not mime_type:
        if serve_filename.lower().endswith('.mp4'):
            mime_type = 'video/mp4'
        elif serve_filename.lower().endswith('.avi'):
            mime_type = 'video/x-msvideo'
        else:
            mime_type = 'application/octet-stream'
    
    print(f"Serving video: {filename}, size: {file_size}, mime: {mime_type}")
    
    return send_from_directory(
        serve_dir, 
        serve_filename,
        mimetype=mime_type,
        as_attachment=False,
        conditional=True  # Enable conditional requests for video streaming
    )

@app.route('/delete_recording/<path:filename>', methods=['DELETE'])
def delete_recording(filename):
    try:
        recordings_dir = os.path.join(os.path.dirname(__file__), 'recordings')
        
        # Handle both direct files and segmented_recordings subdirectory
        if filename.startswith('segmented_recordings/'):
            # Remove the prefix and look in segmented_recordings subdirectory
            actual_filename = filename.replace('segmented_recordings/', '')
            file_path = os.path.join(recordings_dir, 'segmented_recordings', actual_filename)
        else:
            # Look in main recordings directory
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
