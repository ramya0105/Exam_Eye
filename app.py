import os
import cv2
import sqlite3
import atexit
import subprocess
from datetime import datetime
from io import BytesIO
from flask import Flask, render_template, request, url_for, send_from_directory, send_file, redirect, flash, session, Response, jsonify
from ultralytics import YOLO
from werkzeug.security import generate_password_hash, check_password_hash
from realtime_webcam import get_monitor
from neo4j_database import init_neo4j, get_graph_data

def local_now_iso():
    return datetime.now().isoformat()


# ---------------------------
# Flask Setup
# ---------------------------
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")

UPLOAD_FOLDER = 'static/uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'}
DATABASE = "exam_integrity.db"

# ---------------------------
# Load YOLO Model
# ---------------------------
yolo_model = YOLO('best.pt')


def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_db_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                full_name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                mobile TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS proctoring_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT NOT NULL,
                duration_sec INTEGER NOT NULL,
                total_frames INTEGER NOT NULL,
                looking_away_pct REAL NOT NULL,
                eyes_closed_pct REAL NOT NULL,
                phone_detected_pct REAL NOT NULL,
                unauthorized_people_pct REAL NOT NULL,
                stop_reason TEXT NOT NULL,
                video_path TEXT,
                video_blob BLOB,
                full_video_path TEXT,
                full_video_blob BLOB,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS proctoring_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                username TEXT NOT NULL,
                event_type TEXT NOT NULL,
                description TEXT NOT NULL,
                event_time_offset REAL NOT NULL,
                evidence_image_path TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES proctoring_sessions(id)
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_proctoring_sessions_username ON proctoring_sessions(username)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_proctoring_sessions_start_time ON proctoring_sessions(start_time)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_proctoring_logs_username ON proctoring_logs(username)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_proctoring_logs_event_type ON proctoring_logs(event_type)")
        # Lightweight migration support for existing DBs created without video_blob.
        columns = [row["name"] for row in conn.execute("PRAGMA table_info(proctoring_sessions)").fetchall()]
        if "video_blob" not in columns:
            conn.execute("ALTER TABLE proctoring_sessions ADD COLUMN video_blob BLOB")
        if "full_video_path" not in columns:
            conn.execute("ALTER TABLE proctoring_sessions ADD COLUMN full_video_path TEXT")
        if "full_video_blob" not in columns:
            conn.execute("ALTER TABLE proctoring_sessions ADD COLUMN full_video_blob BLOB")
        log_columns = [row["name"] for row in conn.execute("PRAGMA table_info(proctoring_logs)").fetchall()]
        if "evidence_image_path" not in log_columns:
            conn.execute("ALTER TABLE proctoring_logs ADD COLUMN evidence_image_path TEXT")
        conn.commit()


init_db()
init_neo4j()  # Initialize Neo4j graph nodes constraints
realtime_monitor = get_monitor()
def _append_rule_log(logs, event_type, description, offset_sec=0.0):
    logs.append(
        {
            "event_type": event_type,
            "description": description,
            "event_time_offset": float(offset_sec or 0.0),
            "evidence_image_path": None,
            "created_at": local_now_iso(),
        }
    )


def _latest_offset(logs, event_type):
    offsets = [float(l.get("event_time_offset", 0.0)) for l in logs if l.get("event_type") == event_type]
    return max(offsets) if offsets else 0.0


def _find_burst_offset(logs, event_type, window_sec=10.0, min_count=3):
    offsets = sorted(
        float(l.get("event_time_offset", 0.0))
        for l in logs
        if l.get("event_type") == event_type and l.get("event_time_offset") is not None
    )
    if len(offsets) < min_count:
        return None

    i = 0
    for j, t in enumerate(offsets):
        while t - offsets[i] > window_sec:
            i += 1
        if (j - i + 1) >= min_count:
            return t
    return None


def apply_rule_addon_logs(session_data):
    if not session_data:
        return session_data

    logs = list(session_data.get("logs", []) or [])

    def _has_event(event_type):
        return any(l.get("event_type") == event_type for l in logs)

    # Rule 1: Hard violations for phone or multiple people.
    if float(session_data.get("phone_detected_pct", 0.0)) > 0.0 or _has_event("phone_detected"):
        _append_rule_log(
            logs,
            "rule_violation",
            "Hard violation: phone detected during session.",
            _latest_offset(logs, "phone_detected"),
        )

    if float(session_data.get("unauthorized_people_pct", 0.0)) > 0.0 or _has_event("multiple_people"):
        _append_rule_log(
            logs,
            "rule_violation",
            "Hard violation: multiple people detected during session.",
            _latest_offset(logs, "multiple_people"),
        )

    # Rule 2: Sustained attention loss (looking away or eyes closed).
    if (
        float(session_data.get("looking_away_pct", 0.0)) >= 20.0
        or float(session_data.get("eyes_closed_pct", 0.0)) >= 10.0
    ):
        _append_rule_log(
            logs,
            "rule_violation",
            "Sustained attention loss: looking away or eyes closed exceeded threshold.",
            max(_latest_offset(logs, "looking_away"), _latest_offset(logs, "eyes_closed")),
        )

    # Rule 3: Burst rule (3+ gaze or looking-away events within 10 seconds).
    for evt in ("gaze_away", "looking_away"):
        burst_offset = _find_burst_offset(logs, evt, window_sec=10.0, min_count=3)
        if burst_offset is not None:
            _append_rule_log(
                logs,
                "rule_violation",
                f"Burst detected: {evt} occurred 3+ times within 10 seconds.",
                burst_offset,
            )
            break

    session_data["logs"] = logs
    return session_data


def save_realtime_session(session_data):
    if not session_data:
        return
    session_data = apply_rule_addon_logs(session_data)

    video_blob = None
    video_path = session_data.get("video_path")
    if video_path:
        abs_video = os.path.join("static", video_path.replace("/", os.sep))
        if os.path.exists(abs_video):
            try:
                with open(abs_video, "rb") as f:
                    video_blob = f.read()
            except Exception:
                video_blob = None

    full_video_blob = None
    full_video_path = session_data.get("full_video_path")
    if full_video_path:
        abs_full_video = os.path.join("static", full_video_path.replace("/", os.sep))
        if os.path.exists(abs_full_video):
            try:
                with open(abs_full_video, "rb") as f:
                    full_video_blob = f.read()
            except Exception:
                full_video_blob = None

    with get_db_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO proctoring_sessions (
                username, start_time, end_time, duration_sec, total_frames,
                looking_away_pct, eyes_closed_pct, phone_detected_pct,
                unauthorized_people_pct, stop_reason, video_path, video_blob,
                full_video_path, full_video_blob, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_data.get("username", "anonymous"),
                session_data.get("start_time", local_now_iso()),
                session_data.get("end_time", local_now_iso()),
                int(session_data.get("duration_sec", 0)),
                int(session_data.get("total_frames", 0)),
                float(session_data.get("looking_away_pct", 0.0)),
                float(session_data.get("eyes_closed_pct", 0.0)),
                float(session_data.get("phone_detected_pct", 0.0)),
                float(session_data.get("unauthorized_people_pct", 0.0)),
                session_data.get("stop_reason", "stopped"),
                video_path,
                video_blob,
                full_video_path,
                full_video_blob,
                local_now_iso(),
            ),
        )
        session_id = cursor.lastrowid

        from neo4j_database import create_session_node, log_malpractice_event_to_neo4j
        
        # Init the Session Node and Student Node in Graph DB
        create_session_node(session_id, session_data.get("username", "anonymous"), session_data.get("start_time", local_now_iso()))

        logs = session_data.get("logs", []) or []
        for log in logs:
            conn.execute(
                """
                INSERT INTO proctoring_logs (
                    session_id, username, event_type, description, event_time_offset, evidence_image_path, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    session_data.get("username", "anonymous"),
                    log.get("event_type", "unknown"),
                    log.get("description", "No description"),
                    float(log.get("event_time_offset", 0.0)),
                    log.get("evidence_image_path"),
                    log.get("created_at", local_now_iso()),
                ),
            )
            
            # Log the cheating event directly to Neo4j to build Graph connections
            if log.get("event_type") != "rule_violation":
                log_malpractice_event_to_neo4j(
                    session_id=session_id,
                    username=session_data.get("username", "anonymous"),
                    event_type=log.get("event_type", "unknown"),
                    description=log.get("description", "No description"),
                    evidence_path=log.get("evidence_image_path") or ""
                )

        conn.commit()


realtime_monitor.set_persist_callback(save_realtime_session)

# ---------------------------
# Helper Function
# ---------------------------
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def render_detection_result(file_url, file_type):
    template_name = 'admin_result.html' if session.get('admin') else 'result.html'
    return render_template(
        template_name,
        file_url=file_url,
        file_type=file_type
    )


# ---------------------------
# Detection Route
# ---------------------------
@app.route('/detect', methods=['POST'])
def detect():

    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']

    if file.filename == '':
        return "No selected file", 400

    if not allowed_file(file.filename):
        return "Unsupported file format", 400

    # Save file
    filename = datetime.now().strftime("%Y%m%d%H%M%S_") + file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    file_ext = filename.rsplit('.', 1)[1].lower()

    # ---------------- IMAGE DETECTION ----------------
    if file_ext in ['png', 'jpg', 'jpeg']:

        results = yolo_model(filepath)

        result_filename = 'result_' + filename
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)

        result_image = results[0].plot(line_width=2, font_size=12)
        cv2.imwrite(result_path, result_image)

        return render_detection_result(
            file_url=url_for('static', filename='uploads/' + result_filename),
            file_type='image'
        )

    # ---------------- VIDEO DETECTION ----------------
    elif file_ext in ['mp4', 'avi', 'mov']:

        cap = cv2.VideoCapture(filepath)

        if not cap.isOpened():
            return "Error opening video file", 500

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps < 1:
            fps = 25

        result_filename = 'result_' + filename
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)

        # --- Use H.264 codec ---
        fourcc = cv2.VideoWriter_fourcc(*'avc1')

        out = cv2.VideoWriter(result_path, fourcc, fps, (width, height))

        if not out.isOpened():
            print("VideoWriter failed to open. Codec issue.")
            return "VideoWriter failed", 500

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (width, height))

            results = yolo_model(frame)
            annotated_frame = results[0].plot()

            out.write(annotated_frame)

        cap.release()
        out.release()

        print("Output video size:", os.path.getsize(result_path))

        return render_detection_result(
            file_url='/uploads/' + result_filename,
            file_type='video'
        )


    return "Something went wrong", 500

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(
        os.path.join('static', 'uploads'), 
        filename,
        mimetype="video/mp4"
    )


# ---------------------------
# Home Page
# ---------------------------

@app.route('/signup', methods=['GET', 'POST'])
@app.route('/register', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        full_name = request.form.get('full_name', '').strip()
        email = request.form.get('email', '').strip().lower()
        mobile = request.form.get('mobile', '').strip()
        password = request.form.get('password', '')

        if not all([username, full_name, email, mobile, password]):
            flash('All fields are required.', 'error')
            return render_template('signup.html')

        password_hash = generate_password_hash(password)

        try:
            with get_db_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO users (username, full_name, email, mobile, password_hash, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (username, full_name, email, mobile, password_hash, datetime.now().isoformat())
                )
                conn.commit()
            flash('Registration successful. Please sign in.', 'success')
            return redirect(url_for('signin'))
        except sqlite3.IntegrityError:
            flash('Username or email already exists.', 'error')
            return render_template('signup.html')

    return render_template('signup.html')


@app.route('/signin', methods=['GET', 'POST'])
def signin():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        with get_db_connection() as conn:
            user = conn.execute(
                "SELECT * FROM users WHERE username = ?",
                (username,)
            ).fetchone()

        if user and check_password_hash(user['password_hash'], password):
            session['user'] = user['username']
            flash(f"Welcome back, {user['full_name']}!", 'success')
            return redirect(url_for('home'))

        flash('Invalid username or password.', 'error')
        return render_template('signin.html')

    return render_template('signin.html')


@app.route('/admin-login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        if username == 'admin' and password == 'admin':
            session['admin'] = True
            session['user'] = 'admin'
            flash('Admin login successful.', 'success')
            return redirect(url_for('admin_dashboard'))

        flash('Invalid admin credentials.', 'error')
        return render_template('admin_login.html')

    return render_template('admin_login.html')


@app.route('/logout')
def logout():
    session.pop('user', None)
    session.pop('admin', None)
    flash('Logged out successfully.', 'success')
    return redirect(url_for('signin'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('home.html')


@app.route('/realtime')
def realtime():
    if 'user' not in session:
        flash('Please sign in to access realtime monitoring.', 'error')
        return redirect(url_for('signin'))
    return render_template('realtime.html')


@app.route('/graph')
def graph_view():
    """Render the Interactive Neo4j Graph Page"""
    if 'user' not in session and 'admin' not in session:
        flash('Please sign in to access the graph view.', 'error')
        return redirect(url_for('signin'))
    return render_template('graph.html')


@app.route('/api/graph')
def api_graph():
    """Return JSON node/edge data from Neo4j for vis.js to render"""
    if 'user' not in session and 'admin' not in session:
        return jsonify({"status": "unauthorized"}), 401
    
    # Check if driver is initialized
    from neo4j_database import driver
    if not driver:
         return jsonify({"error": "Neo4j Database is not connected. Did you start Neo4j Desktop?"}), 503
         
    from neo4j_database import get_graph_data
    return jsonify(get_graph_data())


@app.route('/realtime/feed')
def realtime_feed():
    if 'user' not in session:
        return Response("Unauthorized", status=401)

    username = session.get('user', 'anonymous')
    return Response(
        realtime_monitor.generate_frames(username),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/realtime/scores')
def realtime_scores():
    if 'user' not in session:
        return jsonify({"status": "unauthorized"}), 401
    return jsonify(realtime_monitor.get_scores())


@app.route('/realtime/stop', methods=['POST'])
def realtime_stop():
    if 'user' not in session:
        return jsonify({"status": "unauthorized"}), 401

    realtime_monitor.stop(reason="stopped_by_user")
    return jsonify({"status": "stopped"})


@app.route('/realtime/restart', methods=['POST'])
def realtime_restart():
    if 'user' not in session:
        return jsonify({"status": "unauthorized"}), 401

    realtime_monitor.stop(reason="restarted_by_user")
    return jsonify({"status": "restarted"})


def _is_admin():
    return bool(session.get("admin"))


def _absolute_static_path(rel_path):
    if not rel_path:
        return None
    rel_clean = rel_path.replace("\\", "/").lstrip("/")
    return os.path.abspath(os.path.join(app.root_path, "static", rel_clean))


def _safe_proctoring_media_path(rel_path):
    if not rel_path:
        return None
    rel_clean = os.path.normpath(rel_path.replace("\\", "/")).replace("\\", "/").lstrip("/")
    if rel_clean.startswith("../"):
        return None
    if not rel_clean.startswith("proctoring_data/"):
        return None

    abs_path = _absolute_static_path(rel_clean)
    proctor_root = os.path.abspath(os.path.join(app.root_path, "static", "proctoring_data"))
    if not abs_path.startswith(proctor_root):
        return None
    return abs_path


def _ensure_web_playable_mp4(abs_path):
    if not abs_path.lower().endswith(".mp4"):
        return abs_path

    web_path = abs_path[:-4] + "_web.mp4"
    try:
        needs_transcode = (not os.path.exists(web_path)) or (
            os.path.getmtime(web_path) < os.path.getmtime(abs_path)
        )
        if not needs_transcode:
            return web_path

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            abs_path,
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            web_path,
        ]
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode == 0 and os.path.exists(web_path):
            return web_path
    except Exception:
        pass
    return abs_path


def _build_session_filter(filters):
    clauses = ["1=1"]
    params = []

    if filters["username"]:
        clauses.append("s.username = ?")
        params.append(filters["username"])

    if filters["date_from"]:
        clauses.append("date(s.start_time) >= date(?)")
        params.append(filters["date_from"])

    if filters["date_to"]:
        clauses.append("date(s.start_time) <= date(?)")
        params.append(filters["date_to"])

    if filters["event_type"]:
        clauses.append(
            """
            EXISTS (
                SELECT 1 FROM proctoring_logs l2
                WHERE l2.session_id = s.id AND l2.event_type = ?
            )
            """
        )
        params.append(filters["event_type"])

    return " AND ".join(clauses), params


def _build_log_filter(filters):
    clauses = ["1=1"]
    params = []

    if filters["username"]:
        clauses.append("s.username = ?")
        params.append(filters["username"])

    if filters["date_from"]:
        clauses.append("date(s.start_time) >= date(?)")
        params.append(filters["date_from"])

    if filters["date_to"]:
        clauses.append("date(s.start_time) <= date(?)")
        params.append(filters["date_to"])

    if filters["event_type"]:
        clauses.append("l.event_type = ?")
        params.append(filters["event_type"])

    return " AND ".join(clauses), params


def _admin_reference_data(conn):
    usernames = conn.execute(
        "SELECT DISTINCT username FROM proctoring_sessions ORDER BY username"
    ).fetchall()
    event_types = conn.execute(
        "SELECT DISTINCT event_type FROM proctoring_logs ORDER BY event_type"
    ).fetchall()
    stop_reasons = conn.execute(
        "SELECT DISTINCT stop_reason FROM proctoring_sessions ORDER BY stop_reason"
    ).fetchall()
    return {
        "usernames": [row["username"] for row in usernames],
        "event_types": [row["event_type"] for row in event_types],
        "stop_reasons": [row["stop_reason"] for row in stop_reasons],
    }


def _parse_page(default=1):
    try:
        page = int(request.args.get("page", default))
        return page if page > 0 else default
    except (TypeError, ValueError):
        return default


def _build_pagination(total_count, page, per_page):
    total_pages = max(1, (total_count + per_page - 1) // per_page)
    page = min(page, total_pages)
    return {
        "page": page,
        "per_page": per_page,
        "total_count": total_count,
        "total_pages": total_pages,
        "has_prev": page > 1,
        "has_next": page < total_pages,
        "prev_page": page - 1 if page > 1 else 1,
        "next_page": page + 1 if page < total_pages else total_pages,
    }


@app.route('/admin/dashboard')
def admin_dashboard():
    if not _is_admin():
        flash('Admin access required.', 'error')
        return redirect(url_for('admin_login'))

    with get_db_connection() as conn:
        summary = conn.execute(
            """
            SELECT
                COUNT(*) AS total_sessions,
                COALESCE(AVG(looking_away_pct), 0) AS avg_away,
                COALESCE(AVG(eyes_closed_pct), 0) AS avg_eyes_closed,
                COALESCE(AVG(phone_detected_pct), 0) AS avg_phone,
                COALESCE(AVG(unauthorized_people_pct), 0) AS avg_unauth
            FROM proctoring_sessions s
            """,
        ).fetchone()

        overview = conn.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM proctoring_logs) AS total_logs,
                (SELECT COUNT(*) FROM proctoring_logs WHERE event_type = 'phone_detected') AS phone_events,
                (SELECT COUNT(*) FROM proctoring_logs WHERE event_type = 'multiple_people') AS multi_people_events
            """,
        ).fetchone()

        top_users = conn.execute(
            """
            SELECT s.username, COUNT(*) AS session_count
            FROM proctoring_sessions s
            GROUP BY s.username
            ORDER BY session_count DESC
            LIMIT 10
            """
        ).fetchall()

        event_stats = conn.execute(
            """
            SELECT l.event_type, COUNT(*) AS event_count
            FROM proctoring_logs l
            JOIN proctoring_sessions s ON s.id = l.session_id
            GROUP BY l.event_type
            ORDER BY event_count DESC
            """
        ).fetchall()

    overview_row = overview if overview else {"total_logs": 0, "phone_events": 0, "multi_people_events": 0}

    return render_template(
        "admin_dashboard.html",
        summary=summary,
        overview=overview_row,
        top_users=top_users,
        event_stats=event_stats,
    )


@app.route('/admin/sessions')
def admin_sessions():
    if not _is_admin():
        flash('Admin access required.', 'error')
        return redirect(url_for('admin_login'))

    filters = {
        "username": request.args.get("username", "").strip(),
        "date_from": request.args.get("date_from", "").strip(),
        "date_to": request.args.get("date_to", "").strip(),
        "stop_reason": request.args.get("stop_reason", "").strip(),
    }

    session_filters = {
        "username": filters["username"],
        "event_type": "",
        "date_from": filters["date_from"],
        "date_to": filters["date_to"],
    }
    where_clause, where_params = _build_session_filter(session_filters)

    if filters["stop_reason"]:
        where_clause += " AND s.stop_reason = ?"
        where_params = where_params + [filters["stop_reason"]]

    per_page = 20
    requested_page = _parse_page()

    with get_db_connection() as conn:
        total_count = conn.execute(
            f"""
            SELECT COUNT(*) AS total_count
            FROM proctoring_sessions s
            WHERE {where_clause}
            """,
            where_params,
        ).fetchone()["total_count"]
        pagination = _build_pagination(total_count, requested_page, per_page)
        offset = (pagination["page"] - 1) * per_page

        sessions = conn.execute(
            f"""
            SELECT s.id, s.username, s.start_time, s.end_time, s.duration_sec, s.total_frames,
                   s.looking_away_pct, s.eyes_closed_pct, s.phone_detected_pct,
                   s.unauthorized_people_pct, s.stop_reason, s.video_path, s.full_video_path, s.created_at,
                   (SELECT COUNT(*) FROM proctoring_logs l WHERE l.session_id = s.id) AS log_count
            FROM proctoring_sessions s
            WHERE {where_clause}
            ORDER BY s.id DESC
            LIMIT ? OFFSET ?
            """,
            where_params + [per_page, offset],
        ).fetchall()

        session_summary = conn.execute(
            f"""
            SELECT
                COUNT(*) AS total_sessions,
                COALESCE(AVG(duration_sec), 0) AS avg_duration,
                COALESCE(AVG(total_frames), 0) AS avg_frames
            FROM proctoring_sessions s
            WHERE {where_clause}
            """,
            where_params,
        ).fetchone()

        refs = _admin_reference_data(conn)

    return render_template(
        "admin_sessions.html",
        filters=filters,
        sessions=sessions,
        session_summary=session_summary,
        usernames=refs["usernames"],
        stop_reasons=refs["stop_reasons"],
        pagination=pagination,
    )


@app.route('/admin/logs')
def admin_logs():
    if not _is_admin():
        flash('Admin access required.', 'error')
        return redirect(url_for('admin_login'))

    filters = {
        "username": request.args.get("username", "").strip(),
        "event_type": request.args.get("event_type", "").strip(),
        "date_from": request.args.get("date_from", "").strip(),
        "date_to": request.args.get("date_to", "").strip(),
        "session_id": request.args.get("session_id", "").strip(),
    }

    log_filters = {
        "username": filters["username"],
        "event_type": filters["event_type"],
        "date_from": filters["date_from"],
        "date_to": filters["date_to"],
    }
    log_where_clause, log_where_params = _build_log_filter(log_filters)

    if filters["session_id"]:
        log_where_clause += " AND l.session_id = ?"
        log_where_params = log_where_params + [filters["session_id"]]

    per_page = 25
    requested_page = _parse_page()

    with get_db_connection() as conn:
        total_count = conn.execute(
            f"""
            SELECT COUNT(*) AS total_count
            FROM proctoring_logs l
            JOIN proctoring_sessions s ON s.id = l.session_id
            WHERE {log_where_clause}
            """,
            log_where_params,
        ).fetchone()["total_count"]
        pagination = _build_pagination(total_count, requested_page, per_page)
        offset = (pagination["page"] - 1) * per_page

        logs = conn.execute(
            f"""
            SELECT l.*, s.start_time
            FROM proctoring_logs l
            JOIN proctoring_sessions s ON s.id = l.session_id
            WHERE {log_where_clause}
            ORDER BY l.id DESC
            LIMIT ? OFFSET ?
            """,
            log_where_params + [per_page, offset],
        ).fetchall()

        log_summary = conn.execute(
            f"""
            SELECT
                COUNT(*) AS total_logs,
                COUNT(DISTINCT l.session_id) AS affected_sessions
            FROM proctoring_logs l
            JOIN proctoring_sessions s ON s.id = l.session_id
            WHERE {log_where_clause}
            """,
            log_where_params,
        ).fetchone()

        top_users = conn.execute(
            f"""
            SELECT s.username, COUNT(*) AS log_count
            FROM proctoring_logs l
            JOIN proctoring_sessions s ON s.id = l.session_id
            WHERE {log_where_clause}
            GROUP BY s.username
            ORDER BY log_count DESC
            LIMIT 10
            """,
            log_where_params,
        ).fetchall()

        event_stats = conn.execute(
            f"""
            SELECT l.event_type, COUNT(*) AS event_count
            FROM proctoring_logs l
            JOIN proctoring_sessions s ON s.id = l.session_id
            WHERE {log_where_clause}
            GROUP BY l.event_type
            ORDER BY event_count DESC
            """,
            log_where_params,
        ).fetchall()

        refs = _admin_reference_data(conn)

    return render_template(
        "admin_logs.html",
        filters=filters,
        logs=logs,
        log_summary=log_summary,
        top_users=top_users,
        event_stats=event_stats,
        usernames=refs["usernames"],
        event_types=refs["event_types"],
        pagination=pagination,
    )


@app.route('/admin/video/<int:session_id>')
def admin_video(session_id):
    if not _is_admin():
        flash('Admin access required.', 'error')
        return redirect(url_for('admin_login'))

    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT video_blob, video_path FROM proctoring_sessions WHERE id = ?",
            (session_id,),
        ).fetchone()

    if not row:
        return "Video not found", 404

    if row["video_path"]:
        abs_path = _absolute_static_path(row["video_path"])
        if abs_path and os.path.exists(abs_path):
            abs_path = _ensure_web_playable_mp4(abs_path)
            return send_file(abs_path, mimetype="video/mp4", conditional=True)

    if row["video_blob"]:
        return send_file(
            BytesIO(row["video_blob"]),
            mimetype="video/mp4",
            as_attachment=False,
            download_name=f"session_{session_id}_segment.mp4",
        )

    return "Video not available", 404


@app.route('/admin/full-video/<int:session_id>')
def admin_full_video(session_id):
    if not _is_admin():
        flash('Admin access required.', 'error')
        return redirect(url_for('admin_login'))

    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT full_video_blob, full_video_path FROM proctoring_sessions WHERE id = ?",
            (session_id,),
        ).fetchone()

    if not row:
        return "Full video not found", 404

    if row["full_video_path"]:
        abs_path = _absolute_static_path(row["full_video_path"])
        if abs_path and os.path.exists(abs_path):
            abs_path = _ensure_web_playable_mp4(abs_path)
            return send_file(abs_path, mimetype="video/mp4", conditional=True)

    if row["full_video_blob"]:
        return send_file(
            BytesIO(row["full_video_blob"]),
            mimetype="video/mp4",
            as_attachment=False,
            download_name=f"session_{session_id}_full.mp4",
        )

    return "Full video not available", 404


@app.route('/admin/media/<path:rel_path>')
def admin_media(rel_path):
    if not _is_admin():
        flash('Admin access required.', 'error')
        return redirect(url_for('admin_login'))

    abs_path = _safe_proctoring_media_path(rel_path)
    if not abs_path or not os.path.exists(abs_path):
        return "Media not found", 404

    ext = os.path.splitext(abs_path)[1].lower()
    if ext == ".mp4":
        mimetype = "video/mp4"
        abs_path = _ensure_web_playable_mp4(abs_path)
    elif ext in (".jpg", ".jpeg"):
        mimetype = "image/jpeg"
    elif ext == ".png":
        mimetype = "image/png"
    else:
        mimetype = "application/octet-stream"

    return send_file(abs_path, mimetype=mimetype, conditional=True)


@atexit.register
def _cleanup_realtime():
    realtime_monitor.stop(reason="app_shutdown")

# ---------------------------
# Run App
# --------------------------
if __name__ == '__main__':
    app.run(debug=False)


