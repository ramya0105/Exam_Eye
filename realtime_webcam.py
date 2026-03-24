import math
import os
import time
from datetime import datetime
from threading import Lock

import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO


LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LEFT_IRIS = [468, 469, 470, 471]
RIGHT_IRIS = [473, 474, 475, 476]
LEFT_EYE_CORNER = [33, 133]
RIGHT_EYE_CORNER = [362, 263]

EAR_THRESHOLD = 0.19
MAX_YAW_OFFSET = 110 * 1.35
MAX_PITCH_OFFSET = 140 * 1.35
EVENT_COOLDOWN_SEC = 1.5
RECORD_FPS = 5.0

MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0),
    ],
    dtype=np.float64,
)


def eye_aspect_ratio(landmarks, eye_indices, w, h):
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_indices]
    a = math.dist(pts[1], pts[5])
    b = math.dist(pts[2], pts[4])
    c = math.dist(pts[0], pts[3])
    return (a + b) / (2.0 * c)


def detect_gaze(landmarks, w, h, eye_corners, iris_points):
    left_corner = np.array(
        [landmarks[eye_corners[0]].x * w, landmarks[eye_corners[0]].y * h]
    )
    right_corner = np.array(
        [landmarks[eye_corners[1]].x * w, landmarks[eye_corners[1]].y * h]
    )

    iris_coords = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in iris_points])
    iris_center = np.mean(iris_coords, axis=0)

    eye_width = right_corner[0] - left_corner[0] + 1e-6
    x_ratio = (iris_center[0] - left_corner[0]) / eye_width

    top_y = min([landmarks[i].y * h for i in iris_points])
    bottom_y = max([landmarks[i].y * h for i in iris_points])
    y_ratio = (iris_center[1] - top_y) / (bottom_y - top_y + 1e-6)

    gaze = "CENTER"
    if x_ratio < 0.30:
        gaze = "LEFT"
    elif x_ratio > 0.70:
        gaze = "RIGHT"
    elif y_ratio < 0.25:
        gaze = "UP"
    elif y_ratio > 0.75:
        gaze = "DOWN"
    return gaze, iris_center.astype(int)


def estimate_head_pose(landmarks, width, height):
    image_points = np.array(
        [
            (landmarks[1].x * width, landmarks[1].y * height),
            (landmarks[152].x * width, landmarks[152].y * height),
            (landmarks[33].x * width, landmarks[33].y * height),
            (landmarks[263].x * width, landmarks[263].y * height),
            (landmarks[61].x * width, landmarks[61].y * height),
            (landmarks[291].x * width, landmarks[291].y * height),
        ],
        dtype=np.float64,
    )

    focal = width
    cam_matrix = np.array(
        [[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], dtype=np.float64
    )

    success, rot_vec, _ = cv2.solvePnP(MODEL_POINTS, image_points, cam_matrix, np.zeros((4, 1)))
    if not success:
        return None

    rmat, _ = cv2.Rodrigues(rot_vec)
    angles, *_ = cv2.RQDecomp3x3(rmat)
    return angles


def local_now_iso():
    return datetime.now().isoformat()


class RealtimeMonitor:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            refine_landmarks=True,
            max_num_faces=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.mp_face_detection = mp.solutions.face_detection.FaceDetection()
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles
        self.yolo = YOLO("yolov8n.pt")

        self.lock = Lock()
        self.cap = None
        self.running = False
        self.persist_callback = None

        self.base_dir = os.path.join("static", "proctoring_data")
        os.makedirs(self.base_dir, exist_ok=True)

        self.away_count = 0
        self.eye_closed_count = 0
        self.phone_detected_count = 0
        self.unauthorized_person_count = 0
        self.total_frames = 0
        self.calibrated_pitch = 0.0
        self.calibrated_yaw = 0.0
        self.start_time = None
        self.started_at_iso = None
        self.active_username = "anonymous"
        self.session_logs = []
        self.last_event_time = {}

        self.segment_writer = None
        self.segment_video_relpath = None
        self.video_fps = RECORD_FPS
        self.video_size = None

        self.full_writer = None
        self.full_video_relpath = None
        self.full_video_username = None
        self.full_session_key = None

    def set_persist_callback(self, callback):
        self.persist_callback = callback

    def _safe_username(self):
        raw = self.active_username or "anonymous"
        cleaned = "".join(c for c in raw if c.isalnum() or c in ("_", "-")).strip()
        return cleaned if cleaned else "anonymous"

    def _user_dirs(self):
        user = self._safe_username()
        user_root = os.path.join(self.base_dir, user)
        dirs = {
            "root": user_root,
            "segments": os.path.join(user_root, "videos", "segments"),
            "full": os.path.join(user_root, "videos", "full"),
            "logs": os.path.join(user_root, "logs"),
        }
        for d in dirs.values():
            os.makedirs(d, exist_ok=True)
        return dirs

    def _reset_metrics(self, username):
        self.away_count = 0
        self.eye_closed_count = 0
        self.phone_detected_count = 0
        self.unauthorized_person_count = 0
        self.total_frames = 0
        self.calibrated_pitch = 0.0
        self.calibrated_yaw = 0.0
        self.start_time = time.time()
        self.started_at_iso = local_now_iso()
        self.active_username = username or "anonymous"
        self.session_logs = []
        self.last_event_time = {}
        self.segment_writer = None
        self.segment_video_relpath = None
        self.video_size = None

    def _segment_filename(self):
        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"segment_{self._safe_username()}_{ts}.mp4"

    def _full_filename(self):
        if not self.full_session_key:
            self.full_session_key = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"full_{self._safe_username()}_{self.full_session_key}.mp4"

    def _open_mp4_writer(self, output_path, frame_size):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, self.video_fps, frame_size)
        if writer.isOpened():
            return writer
        return None

    def _ensure_writers(self, frame):
        h, w, _ = frame.shape
        frame_size = (w, h)
        self.video_size = frame_size
        dirs = self._user_dirs()

        if self.segment_writer is None:
            seg_name = self._segment_filename()
            seg_path = os.path.join(dirs["segments"], seg_name)
            seg_writer = self._open_mp4_writer(seg_path, frame_size)
            if seg_writer is not None:
                self.segment_writer = seg_writer
                self.segment_video_relpath = os.path.relpath(seg_path, "static").replace("\\", "/")

        if self.full_video_username != self.active_username:
            if self.full_writer is not None:
                self.full_writer.release()
            self.full_writer = None
            self.full_video_relpath = None
            self.full_video_username = self.active_username
            self.full_session_key = None

        if self.full_writer is None:
            full_name = self._full_filename()
            full_path = os.path.join(dirs["full"], full_name)
            full_writer = self._open_mp4_writer(full_path, frame_size)
            if full_writer is not None:
                self.full_writer = full_writer
                self.full_video_relpath = os.path.relpath(full_path, "static").replace("\\", "/")

    def _event_image_relpath(self, event_type):
        ts = datetime.now().strftime("%Y%m%d%H%M%S%f")
        safe_event = "".join(c for c in event_type if c.isalnum() or c in ("_", "-")).strip()
        if not safe_event:
            safe_event = "event"
        filename = f"log_{safe_event}_{ts}.jpg"
        log_path = os.path.join(self._user_dirs()["logs"], filename)
        return os.path.relpath(log_path, "static").replace("\\", "/")

    def _record_event(self, event_type, description, frame=None):
        now = time.time()
        last = self.last_event_time.get(event_type, 0.0)
        if now - last < EVENT_COOLDOWN_SEC:
            return
        self.last_event_time[event_type] = now

        offset_sec = 0.0
        if self.start_time is not None:
            offset_sec = round(now - self.start_time, 2)

        image_relpath = None
        if frame is not None:
            image_relpath = self._event_image_relpath(event_type)
            image_abspath = os.path.join("static", image_relpath.replace("/", os.sep))
            try:
                cv2.imwrite(image_abspath, frame)
            except Exception:
                image_relpath = None

        self.session_logs.append(
            {
                "event_type": event_type,
                "description": description,
                "event_time_offset": offset_sec,
                "evidence_image_path": image_relpath,
                "created_at": local_now_iso(),
            }
        )

    def start(self, username):
        with self.lock:
            if self.running:
                return

            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.cap = None
                raise RuntimeError("Unable to open webcam.")

            # Use a stable writer FPS close to the processed frame throughput.
            # This avoids fast playback when model inference is slower than camera FPS.
            self.video_fps = RECORD_FPS

            self._reset_metrics(username)
            if self.full_video_username != (username or "anonymous"):
                self.full_video_username = username or "anonymous"
                self.full_session_key = None
            self._calibrate()
            self.running = True

    def _finalize_session_data(self, stop_reason):
        ended_at = local_now_iso()
        duration = int(time.time() - self.start_time) if self.start_time else 0
        full_video_path = self.full_video_relpath if stop_reason != "restarted_by_user" else None
        return {
            "username": self.active_username,
            "start_time": self.started_at_iso,
            "end_time": ended_at,
            "duration_sec": duration,
            "total_frames": self.total_frames,
            "looking_away_pct": self._percent(self.away_count),
            "eyes_closed_pct": self._percent(self.eye_closed_count),
            "phone_detected_pct": self._percent(self.phone_detected_count),
            "unauthorized_people_pct": self._percent(self.unauthorized_person_count),
            "stop_reason": stop_reason,
            "video_path": self.segment_video_relpath,
            "full_video_path": full_video_path,
            "logs": list(self.session_logs),
        }

    def _clear_segment_state(self):
        self.start_time = None
        self.started_at_iso = None
        self.active_username = "anonymous"
        self.session_logs = []
        self.last_event_time = {}
        self.total_frames = 0
        self.away_count = 0
        self.eye_closed_count = 0
        self.phone_detected_count = 0
        self.unauthorized_person_count = 0
        self.segment_video_relpath = None
        self.video_size = None

    def stop(self, reason="stopped"):
        session_data = None
        callback = self.persist_callback

        with self.lock:
            if self.segment_writer is not None:
                self.segment_writer.release()
                self.segment_writer = None

            if self.cap is not None:
                self.cap.release()
                self.cap = None

            if self.running or self.total_frames > 0:
                session_data = self._finalize_session_data(reason)

            if reason != "restarted_by_user":
                if self.full_writer is not None:
                    self.full_writer.release()
                self.full_writer = None
                self.full_video_relpath = None
                self.full_video_username = None
                self.full_session_key = None

            self.running = False
            self._clear_segment_state()

        if callback and session_data is not None:
            try:
                callback(session_data)
            except Exception:
                pass

        return session_data

    def _calibrate(self):
        if self.cap is None:
            return

        calibration_data = []
        start_cal = time.time()
        while time.time() - start_cal < 2.0:
            ret, frame = self.cap.read()
            if not ret:
                continue

            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mp_face_mesh.process(rgb)
            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                angles = estimate_head_pose(lm, w, h)
                if angles is not None:
                    calibration_data.append((angles[0], angles[1]))

        if calibration_data:
            self.calibrated_pitch = float(np.mean([p[0] for p in calibration_data]))
            self.calibrated_yaw = float(np.mean([p[1] for p in calibration_data]))

    def _is_looking_away(self, pitch, yaw):
        return (
            abs(pitch - self.calibrated_pitch) > MAX_PITCH_OFFSET
            or abs(yaw - self.calibrated_yaw) > MAX_YAW_OFFSET
        )

    def _annotate_frame(self, frame):
        self.total_frames += 1
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results_mesh = self.mp_face_mesh.process(rgb)
        results_faces = self.mp_face_detection.process(rgb)

        if results_mesh.multi_face_landmarks:
            for face_landmarks in results_mesh.multi_face_landmarks:
                self.mp_draw.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_styles.get_default_face_mesh_tesselation_style(),
                )
                self.mp_draw.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_draw.DrawingSpec(
                        color=(0, 255, 0), thickness=1, circle_radius=1
                    ),
                )

            lm = results_mesh.multi_face_landmarks[0].landmark
            angles = estimate_head_pose(lm, w, h)
            if angles is not None:
                pitch, yaw, _ = angles
                if self._is_looking_away(pitch, yaw):
                    self.away_count += 1
                    self._record_event(
                        "looking_away",
                        "Candidate head pose deviated from calibrated position.",
                        frame,
                    )
                    cv2.putText(frame, "LOOKING AWAY!", (25, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            left_ear = eye_aspect_ratio(lm, LEFT_EYE, w, h)
            right_ear = eye_aspect_ratio(lm, RIGHT_EYE, w, h)
            ear = (left_ear + right_ear) / 2.0
            if ear < EAR_THRESHOLD:
                self.eye_closed_count += 1
                self._record_event(
                    "eyes_closed",
                    "Eye aspect ratio dropped below threshold, possible drowsiness.",
                    frame,
                )
                cv2.putText(frame, "EYES CLOSED / SLEEPY!", (25, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            gaze_left, iris_l = detect_gaze(lm, w, h, LEFT_EYE_CORNER, LEFT_IRIS)
            gaze_right, iris_r = detect_gaze(lm, w, h, RIGHT_EYE_CORNER, RIGHT_IRIS)
            gaze_output = gaze_left if gaze_left == gaze_right else "UNCERTAIN"

            cv2.circle(frame, tuple(iris_l), 3, (0, 0, 255), -1)
            cv2.circle(frame, tuple(iris_r), 3, (0, 0, 255), -1)
            cv2.putText(frame, f"GAZE: {gaze_output}", (25, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            if gaze_output != "CENTER":
                self._record_event(
                    "gaze_away",
                    f"Detected non-center gaze direction: {gaze_output}.",
                    frame,
                )
                cv2.putText(frame, "EYE LOOKING AWAY!", (25, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        if results_faces.detections and len(results_faces.detections) > 1:
            self.unauthorized_person_count += 1
            self._record_event(
                "multiple_people",
                "More than one face was detected in the monitoring frame.",
                frame,
            )
            cv2.putText(frame, "MULTIPLE PEOPLE DETECTED!", (25, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        phone_found = False
        yolo_results = self.yolo(frame, stream=True, verbose=False)
        for r in yolo_results:
            for box in r.boxes:
                cls = r.names[int(box.cls[0])]
                if cls == "cell phone":
                    phone_found = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        if phone_found:
            self.phone_detected_count += 1
            self._record_event("phone_detected", "Cell phone object detected during proctoring session.", frame)
            cv2.putText(frame, "PHONE DETECTED!", (25, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        self._ensure_writers(frame)
        if self.segment_writer is not None:
            self.segment_writer.write(frame)
        if self.full_writer is not None:
            self.full_writer.write(frame)

        return frame

    def generate_frames(self, username):
        self.start(username)
        try:
            while True:
                with self.lock:
                    if not self.running or self.cap is None:
                        break
                    ret, frame = self.cap.read()

                if not ret:
                    continue

                frame = self._annotate_frame(frame)
                ok, buffer = cv2.imencode(".jpg", frame)
                if not ok:
                    continue

                jpg = buffer.tobytes()
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
        finally:
            if self.running:
                self.stop(reason="stream_disconnected")

    def _percent(self, count):
        if self.total_frames <= 0:
            return 0.0
        return round((count / self.total_frames) * 100, 2)

    def get_scores(self):
        duration = int(time.time() - self.start_time) if self.start_time else 0
        return {
            "running": self.running,
            "username": self.active_username,
            "duration_sec": duration,
            "total_frames": self.total_frames,
            "looking_away_pct": self._percent(self.away_count),
            "eyes_closed_pct": self._percent(self.eye_closed_count),
            "phone_detected_pct": self._percent(self.phone_detected_count),
            "unauthorized_people_pct": self._percent(self.unauthorized_person_count),
            "log_count": len(self.session_logs),
        }


_monitor = RealtimeMonitor()


def get_monitor():
    return _monitor
