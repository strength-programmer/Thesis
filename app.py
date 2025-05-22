from flask import Flask, render_template, Response, jsonify, request, send_from_directory
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))
import livefeed

app = Flask(__name__)

# Start the background video thread when the app starts
livefeed.start_background_video_thread()


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/livefeed")
def livefeed_page():
    return render_template("livefeed.html")


@app.route("/recordings")
def recordings():
    return render_template("recordings.html")


@app.route("/employees")
def employees():
    return render_template("employees.html")


@app.route("/account")
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


if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5050)
