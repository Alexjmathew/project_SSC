from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, session
import cv2
import mediapipe as mp
import numpy as np
import time
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore
import google.generativeai as genai

# ── Gemini Configuration ──────────────────────────────────────────────────────
GEMINI_API_KEY = ""  # <-- Replace with your key
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash")
# ─────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Initialize Firebase
cred = credentials.Certificate('firebase.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

# Mediapipe and utility setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Global variables
count = 0
target_count = 0
position = None
exercise_started = False
feedback_message = "Begin PostureTraining!"
start_time = None
last_rep_time = None
exercise = None


# Calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

    return np.degrees(angle)


# Generate video frames
def generate_frames():
    global count, position, exercise_started, feedback_message, start_time, last_rep_time, exercise
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks and exercise_started and exercise:
            landmarks = results.pose_landmarks.landmark

            joints = exercise["joints"]
            coords = [
                [landmarks[getattr(mp_pose.PoseLandmark, joint).value].x,
                 landmarks[getattr(mp_pose.PoseLandmark, joint).value].y]
                for joint in joints
            ]

            angle = calculate_angle(*coords)

            # PostureTraining counting and feedback logic
            if angle > exercise["target_angle"] + exercise["threshold"]:
                position = "up"
            if position == "up" and angle < exercise["target_angle"] - exercise["threshold"]:
                position = "down"
                count += 1

                # Calculate time for the repetition
                current_time = time.time()
                if last_rep_time:
                    rep_time = current_time - last_rep_time
                    if exercise["optimal_speed_range"][0] <= rep_time <= exercise["optimal_speed_range"][1]:
                        feedback_message = "Good speed! Keep going."
                    elif rep_time < exercise["optimal_speed_range"][0]:
                        feedback_message = "Too fast! Slow down."
                    else:
                        feedback_message = "Too slow! Speed up."
                last_rep_time = current_time

                # Start timer for the first rep
                if count == 1:
                    start_time = current_time

            # Provide feedback based on angle
            if angle < exercise["target_angle"] - exercise["threshold"]:
                feedback_message = "Lower your knee slightly"
            elif angle > exercise["target_angle"] + exercise["threshold"]:
                feedback_message = "Raise your knee higher!"

            # Draw feedback on the frame
            cv2.putText(image, f'Angle: {int(angle)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(image, f'Count: {count}/{target_count}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255),
                        2)
            cv2.putText(image, feedback_message, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Stop exercise if target count is reached
            if count >= target_count:
                exercise_started = False
                total_time = time.time() - start_time if start_time else 0
                feedback_message = f"PostureTraining Complete! Total time: {total_time:.2f}s"
                cv2.putText(image, feedback_message, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Save session data
                if 'email' in session:
                    session_data = {
                        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "count": count,
                        "total_time": total_time,
                        "average_speed": total_time / count if count > 0 else 0
                    }
                    user_ref = db.collection('users').document(session['email'])
                    user_ref.update({"sessions": firestore.ArrayUnion([session_data])})

                # Reset global variables for the next session
                start_time = None
                last_rep_time = None
                count = 0

        # Encode the frame and send to the frontend
        ret, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


# Routes
@app.route('/')
def index():
    if 'email' in session:
        return redirect(url_for('profile'))
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        # Query users by email
        users_ref = db.collection('users').where('email', '==', email).limit(1)
        users = list(users_ref.stream())

        if users and users[0].to_dict()['password'] == password:
            session['email'] = email
            session['username'] = users[0].to_dict()['username']
            return redirect(url_for('profile'))
        else:
            return render_template('login.html', error="Invalid email or password.")
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('email', None)
    session.pop('username', None)
    return redirect(url_for('login'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        age = request.form['age']
        height = request.form['height']
        weight = request.form['weight']
        blood_group = request.form['blood_group']

        # Check if email already exists
        email_query = db.collection('users').where('email', '==', email).limit(1)
        if list(email_query.stream()):
            return render_template('register.html', error="Email already registered.")

        # Create document with email as the document ID
        user_ref = db.collection('users').document(email)

        user_data = {
            "username": username,
            "email": email,
            "password": password,
            "age": age,
            "height": height,
            "weight": weight,
            "blood_group": blood_group,
            "sessions": []
        }
        user_ref.set(user_data)
        session['email'] = email
        session['username'] = username
        return redirect(url_for('profile'))
    return render_template('register.html')


@app.route('/profile')
def profile():
    if 'email' not in session:
        return redirect(url_for('login'))

    user_ref = db.collection('users').document(session['email'])
    user = user_ref.get()

    if not user.exists:
        return redirect(url_for('login'))

    user_data = user.to_dict()
    sessions = user_data.get('sessions', [])

    # Prepare data for charts
    session_dates = [session['date'] for session in sessions]
    session_counts = [session['count'] for session in sessions]
    session_total_times = [session['total_time'] for session in sessions]
    session_average_speeds = [session['average_speed'] for session in sessions]

    return render_template('profile.html',
                           user=user_data,
                           session_dates=session_dates,
                           session_counts=session_counts,
                           session_total_times=session_total_times,
                           session_average_speeds=session_average_speeds)


@app.route('/admin')
def admin():
    if 'email' not in session or session['username'] != 'ALEXJMATHEW':
        return redirect(url_for('login'))
    users = list(db.collection('users').stream())
    users = [user.to_dict() for user in users]
    return render_template('admin.html', users=users)


@app.route('/save_session', methods=['POST'])
def save_session():
    global count, start_time, last_rep_time
    if 'email' not in session:
        return jsonify({'success': False, 'error': 'User not logged in'})

    if count == 0:
        return jsonify({'success': False, 'error': 'No exercise data to save'})

    total_time = time.time() - start_time if start_time else 0
    session_data = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "count": count,
        "total_time": total_time,
        "average_speed": total_time / count if count > 0 else 0
    }

    user_ref = db.collection('users').document(session['email'])
    user_ref.update({"sessions": firestore.ArrayUnion([session_data])})

    return jsonify({'success': True})


@app.template_filter('mean')
def mean_filter(values):
    if not values:
        return 0
    return sum(values) / len(values)


# Register the filter
app.jinja_env.filters['mean'] = mean_filter


@app.route('/recommendations')
def recommendations():
    if 'email' not in session:
        return redirect(url_for('login'))

    user_ref = db.collection('users').document(session['email'])
    user = user_ref.get()

    if not user.exists or 'sessions' not in user.to_dict() or len(user.to_dict()['sessions']) == 0:
        return render_template('recommendation.html', error="No session data found.")

    # Calculate average speed
    sessions = user.to_dict()['sessions']
    avg_speed = np.mean([session['average_speed'] for session in sessions if session['average_speed'] > 0])

    # Rule-based recommendations
    recommendations = []
    if avg_speed < 1.5:
        recommendations.append("Your speed is too fast. Focus on controlled movements to improve form.")
    if avg_speed > 3.0:
        recommendations.append("Your speed is too slow. Try to increase your pace for better endurance.")
    if not recommendations:
        recommendations.append("Great job! Keep up the good work and aim for consistency.")

    return render_template('recommendation.html', recommendations=recommendations, user=user.to_dict(),
                           avg_speed=avg_speed)


@app.route('/select_exercise', methods=['GET', 'POST'])
def select_exercise():
    if 'email' not in session:
        return redirect(url_for('login'))
    global exercise
    if request.method == 'POST':
        exercise_name = request.form['exercise']
        if exercise_name == "knee_raises":
            exercise = {
                "joints": ["LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"],
                "target_angle": 60,  # Ideal angle for knee raised to hip level
                "threshold": 15,
                "optimal_speed_range": (1.0, 2.5)  # Optimal time in seconds for one rep
            }
        elif exercise_name == "squats":
            exercise = {
                "joints": ["LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"],
                "target_angle": 90,  # Ideal angle for squat
                "threshold": 15,
                "optimal_speed_range": (2.0, 4.0)  # Optimal time in seconds for one rep
            }
        return redirect(url_for('training'))
    return render_template('select_exercise.html')


@app.route('/training')
def training():
    if 'email' not in session:
        return redirect(url_for('login'))
    return render_template('training.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/get_count')
def get_count():
    global count, target_count, feedback_message
    return jsonify({'count': count, 'target': target_count, 'feedback': feedback_message})


@app.route('/set_target', methods=['POST'])
def set_target():
    global target_count, count, exercise_started, feedback_message, start_time, last_rep_time
    data = request.json
    target_count = int(data.get('target', 0))
    count = 0
    exercise_started = True
    feedback_message = "Begin PostureTraining!"
    start_time = None
    last_rep_time = None
    return jsonify({'success': True, 'target': target_count})


# ── AI Chatbot Routes ─────────────────────────────────────────────────────────
@app.route('/chatbot')
def chatbot():
    if 'email' not in session:
        return redirect(url_for('login'))

    user_ref = db.collection('users').document(session['email'])
    user = user_ref.get()
    user_data = user.to_dict() if user.exists else {}
    sessions = user_data.get('sessions', [])

    # Build a quick stats summary to seed the welcome message
    total_sessions = len(sessions)
    total_reps = sum(s.get('count', 0) for s in sessions)
    avg_speed = (
        sum(s.get('average_speed', 0) for s in sessions) / total_sessions
        if total_sessions > 0 else 0
    )
    last_session = sessions[-1] if sessions else None

    return render_template(
        'chatbot.html',
        user=user_data,
        total_sessions=total_sessions,
        total_reps=total_reps,
        avg_speed=round(avg_speed, 2),
        last_session=last_session
    )


@app.route('/chatbot/message', methods=['POST'])
def chatbot_message():
    if 'email' not in session:
        return jsonify({'error': 'Not logged in'}), 401

    user_message = request.json.get('message', '').strip()
    if not user_message:
        return jsonify({'error': 'Empty message'}), 400

    # Fetch user profile & workout history
    user_ref = db.collection('users').document(session['email'])
    user = user_ref.get()
    user_data = user.to_dict() if user.exists else {}
    sessions_data = user_data.get('sessions', [])

    # Build context summary from workout data
    total_sessions = len(sessions_data)
    total_reps = sum(s.get('count', 0) for s in sessions_data)
    avg_speed = (
        sum(s.get('average_speed', 0) for s in sessions_data) / total_sessions
        if total_sessions > 0 else 0
    )
    last_3 = sessions_data[-3:] if len(sessions_data) >= 3 else sessions_data
    recent_summary = "\n".join(
        f"  - Date: {s.get('date','N/A')}, Reps: {s.get('count',0)}, "
        f"Time: {s.get('total_time',0):.1f}s, Avg Speed: {s.get('average_speed',0):.2f}s/rep"
        for s in last_3
    ) or "  No recent sessions."

    system_prompt = f"""
You are FitBot, an expert AI physiotherapy and fitness coach integrated into a posture-training app.
You have access to the workout history of the user named '{user_data.get('username', 'User')}'.

User Profile:
  - Age: {user_data.get('age', 'N/A')}
  - Height: {user_data.get('height', 'N/A')} cm
  - Weight: {user_data.get('weight', 'N/A')} kg
  - Blood Group: {user_data.get('blood_group', 'N/A')}

Workout Summary:
  - Total sessions completed: {total_sessions}
  - Total reps across all sessions: {total_reps}
  - Average rep speed: {avg_speed:.2f} seconds/rep (lower = faster)

Last {len(last_3)} session(s):
{recent_summary}

Note on speed: The app's optimal rep speed range is 1.0–2.5s for knee raises and 2.0–4.0s for squats.
Analyse the user's data and answer their question helpfully, giving personalised fitness advice.
Be concise, motivating, and professional. Use bullet points where it helps clarity.
"""

    full_prompt = f"{system_prompt}\n\nUser question: {user_message}"

    try:
        response = gemini_model.generate_content(full_prompt)
        reply = response.text
    except Exception as e:
        reply = f"Sorry, I encountered an error: {str(e)}"

    return jsonify({'reply': reply})
# ─────────────────────────────────────────────────────────────────────────────


# ── Manual Exercise Entry Route ───────────────────────────────────────────────
@app.route('/manual_entry', methods=['GET', 'POST'])
def manual_entry():
    if 'email' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        try:
            exercise_name = request.form.get('exercise_name', 'Unknown')
            reps          = int(request.form.get('reps', 0))
            total_time    = float(request.form.get('total_time', 0))
            notes         = request.form.get('notes', '').strip()

            avg_speed = total_time / reps if reps > 0 else 0

            session_data = {
                "date":          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "count":         reps,
                "total_time":    total_time,
                "average_speed": avg_speed,
                "exercise":      exercise_name,
                "notes":         notes,
                "manual":        True
            }

            user_ref = db.collection('users').document(session['email'])
            user_ref.update({"sessions": firestore.ArrayUnion([session_data])})

            return jsonify({'success': True, 'message': 'Session saved successfully!'})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 400

    return render_template('manual_entry.html')
# ─────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    app.run(debug=True)
