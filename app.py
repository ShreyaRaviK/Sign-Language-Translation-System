from flask import Flask, render_template, request, Response, jsonify, send_from_directory
import os
import cv2
import numpy as np
import mediapipe as mp
import pickle
import json
from collections import deque, Counter
import threading
import time
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import hashlib
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

try:
    from google import genai
except ImportError as e:
    raise ImportError(
        "Gemini SDK import failed. Run with the project venv interpreter:\n"
        "  .\\venv\\Scripts\\python.exe -m pip install -U google-genai\n"
        "  .\\venv\\Scripts\\python.exe app.py\n"
        f"Current interpreter: {sys.executable}"
    ) from e
from gtts import gTTS

app = Flask(__name__)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

VIDEO_CACHE_PATH = "video_cache.json"
GEMINI_CACHE_PATH = "gemini_cache.json"

def _load_json_cache(path):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"Failed to load cache {path}: {e}")
    return {}


def _save_json_cache(cache, path):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        print(f"Failed to save cache {path}: {e}")


def _video_cache_key(video_hash, mode):
    return f"{video_hash}:{mode}"


def _normalize_cache_key(words):
    return " ".join((words or "").strip().split()).upper()


def _hash_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def save_uploaded_video(file):
    safe_name = secure_filename(file.filename) or "upload"
    temp_name = f"tmp_{int(time.time() * 1000)}_{safe_name}"
    temp_path = os.path.join(UPLOAD_FOLDER, temp_name)
    file.save(temp_path)

    video_hash = _hash_file(temp_path)
    _, ext = os.path.splitext(safe_name)
    ext = ext.lower() or ".mp4"
    final_name = f"{video_hash}{ext}"
    final_path = os.path.join(UPLOAD_FOLDER, final_name)

    if os.path.exists(final_path):
        os.remove(temp_path)
    else:
        os.replace(temp_path, final_path)

    return final_path, video_hash, f"/uploads/{final_name}"


VIDEO_CACHE = _load_json_cache(VIDEO_CACHE_PATH)
GEMINI_CACHE = _load_json_cache(GEMINI_CACHE_PATH)

# Live mode configuration/state
VALID_MODES = {"letter", "number", "word"}
LIVE_STATE = {
    "letter": {"pending": None, "confirmed": [], "streak_label": None, "streak_count": 0, "last_added": None, "last_added_at": 0.0},
    "number": {"pending": None, "confirmed": [], "streak_label": None, "streak_count": 0, "last_added": None, "last_added_at": 0.0},
    "word": {"pending": None, "confirmed": [], "streak_label": None, "streak_count": 0, "last_added": None, "last_added_at": 0.0},
}
LIVE_LOCK = threading.Lock()
AUTO_ACCEPT_MIN_STREAK = {"letter": 3, "number": 3, "word": 4}
AUTO_ACCEPT_COOLDOWN_SEC = {"letter": 0.6, "number": 0.6, "word": 1.0}

COMBINED_OUTPUT = []
LAST_GEMINI_ERROR = None

# Confidence thresholds for letter/number classifiers
LETTER_CONFIDENCE_THRESHOLD = 0.1
NUMBER_CONFIDENCE_THRESHOLD = 0.75

def _unwrap_model(obj, name):
    if hasattr(obj, "predict"):
        return obj
    if isinstance(obj, dict):
        for key in ("model", "clf", "classifier", "estimator"):
            candidate = obj.get(key)
            if hasattr(candidate, "predict"):
                return candidate
        keys = ", ".join(obj.keys())
        raise TypeError(f"{name} is a dict without a predict() model. Keys: {keys}")
    raise TypeError(f"{name} does not implement predict(). Type: {type(obj)}")

# Load models
letter_model_raw = pickle.load(open("models/classify_letter_model.p", "rb"))
number_model_raw = pickle.load(open("models/classify_number_model.p", "rb"))
letter_model = _unwrap_model(letter_model_raw, "letter_model")
number_model = _unwrap_model(number_model_raw, "number_model")

with open("models/label_map.json") as f:
    label_map = {int(k): v for k, v in json.load(f).items()}

# ==============================
# NEW WORD MODEL CONFIGURATION
# ==============================
WORD_MODEL_PATH = "models/final_model.pt"
WORD_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 126
INPUT_CHANNELS = 21 
SEQ_LEN = 64
POSE_KEEP = [0, 9, 10, 11, 12, 13, 14, 15, 16, 23, 24]

# Sliding window config
WINDOW_SIZE = 32  # Reduced from 64 for faster processing            
STRIDE = 16  # Reduced from 32
CONFIDENCE_THRESHOLD = 0.7

# --- MODEL ARCHITECTURE ---
class Block(nn.Module):
    def __init__(self, c1, c2, adj):
        super().__init__()
        self.A = nn.Parameter(adj.clone())
        self.gcn = nn.Conv2d(c1, c2, 1)
        self.tcn = nn.Conv2d(c2, c2, (15, 1), padding=(7, 0))
        self.bn = nn.BatchNorm2d(c2)
        self.drop = nn.Dropout(0.5)
        self.res = nn.Conv2d(c1, c2, 1) if c1 != c2 else nn.Identity()

    def forward(self, x):
        res = self.res(x)
        x = torch.einsum("bctn,nm->bctm", x, self.A)
        x = self.gcn(x)
        x = self.tcn(x)
        x = self.bn(x)
        return self.drop(F.relu(x + res))

class Model(nn.Module):
    def __init__(self, num_classes, adj):
        super().__init__()
        self.b1 = Block(INPUT_CHANNELS, 64, adj)
        self.b2 = Block(64, 128, adj)
        self.b3 = Block(128, 128, adj)
        self.b4 = Block(128, 256, adj)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.embed = nn.Sequential(nn.Flatten(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 128))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.pool(x)
        emb = self.embed(x)
        return self.fc(emb)

def get_adjacency():
    edges = [
        (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20),(0,21),(21,22),(22,23),(23,24),
        (0,25),(25,26),(26,27),(27,28),(0,29),(29,30),(30,31),(31,32),
        (0,33),(33,34),(34,35),(35,36),(0,37),(37,38),(38,39),(39,40),
        (0,41),(41,42),(42,43),(43,44),(0,45),(45,46),(46,47),(47,48),
        (0,49),(49,50),(50,51),(51,52),
    ]
    A = np.eye(53)
    for i, j in edges:
        A[i, j] = 1
        A[j, i] = 1
    deg = A.sum(1)
    D = np.diag(1 / np.sqrt(deg))
    return torch.tensor(D @ A @ D, dtype=torch.float32)

adj = get_adjacency().to(WORD_DEVICE)
word_model = Model(NUM_CLASSES, adj).to(WORD_DEVICE)
word_model.load_state_dict(torch.load(WORD_MODEL_PATH, map_location=WORD_DEVICE))
word_model.eval()

print("final_model.pt loaded successfully")

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

# --- WORD PROCESSING UTILS ---
def sample_frame_indices(n_frames, seq_len):
    if n_frames >= seq_len:
        return np.linspace(0, n_frames-1, seq_len).astype(int)
    else:
        base = np.arange(n_frames)
        pad = np.full(seq_len - n_frames, n_frames-1)
        return np.concatenate([base, pad])

def format_tensor_chunk(nodes_chunk):
    """Applies the coordinate/math quirks to a specific window of frames."""
    idx = sample_frame_indices(len(nodes_chunk), SEQ_LEN)
    nodes = nodes_chunk[idx] # (64, 53, 4)

    try:
        ls = POSE_KEEP.index(11)
        rs = POSE_KEEP.index(12)
        centers = (nodes[:, ls, :3] + nodes[:, rs, :3]) / 2.0
        scale = np.mean(np.linalg.norm(nodes[:, ls, :2] - nodes[:, rs, :2], axis=-1)) + 1e-6
    except:
        centers = np.mean(nodes[:, :, :3], axis=1)
        scale = 1.0

    coords = (nodes[..., :3] - centers[:, None, :]) / scale
    vis = nodes[..., 3:4]
    vel = np.zeros_like(coords)
    vel[1:] = coords[1:] - coords[:-1]
    x_base = np.concatenate([coords, vis, vel], axis=-1).astype(np.float32)

    x = torch.tensor(x_base).float() # [64, 53, 7]
    
    root = x[:, :, 0:1] 
    x[:, :, :3] -= root[:, :, :3]

    velocity = x[:, 1:, :] - x[:, :-1, :]
    velocity = torch.cat([velocity[:, :1, :], velocity], dim=1)

    bone = x.clone()
    bone[:, :, 1:] = x[:, :, 1:] - x[:, :, :-1]

    x = torch.cat([x, velocity, bone], dim=-1) # (64, 53, 21)

    mask = (vis > 0).astype(np.float32) # (64, 53, 1)
    mask_tensor = torch.tensor(mask)
    x = x * mask_tensor 

    return x.permute(2, 0, 1).unsqueeze(0) # (1, 21, 64, 53)

def extract_all_keypoints(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if not frames: return None

    pose_out, hands_out = [], []
    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5) as pose_mod, \
         mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5) as hands_mod:
        for img in frames:
            res_p = pose_mod.process(img)
            res_h = hands_mod.process(img)

            p = np.zeros((33, 4), dtype=np.float32)
            if res_p and res_p.pose_landmarks:
                for i, lm in enumerate(res_p.pose_landmarks.landmark):
                    p[i] = [lm.x, lm.y, lm.z, getattr(lm, "visibility", 1.0)]

            h = np.zeros((2, 21, 4), dtype=np.float32)
            if res_h and res_h.multi_hand_landmarks:
                for i, hland in enumerate(res_h.multi_hand_landmarks):
                    label = res_h.multi_handedness[i].classification[0].label
                    hand_idx = 0 if label == 'Left' else 1
                    for j, lm in enumerate(hland.landmark):
                        h[hand_idx, j] = [lm.x, lm.y, lm.z, 1.0]

            pose_out.append(p)
            hands_out.append(h)

    pose_out = np.stack(pose_out)
    hands_out = np.stack(hands_out)

    pose_sel = pose_out[:, POSE_KEEP, :]
    left = hands_out[:, 0, :, :]
    right = hands_out[:, 1, :, :]
    nodes = np.concatenate([pose_sel, left, right], axis=1) # (Total_Frames, 53, 4)
    return nodes

def process_word(video_path):
    all_nodes = extract_all_keypoints(video_path)
    if all_nodes is None:
        return None, "Failed to process video."

    total_frames = len(all_nodes)
    raw_predictions = []

    if total_frames <= WINDOW_SIZE + (STRIDE // 2):
        chunks = [all_nodes]
    else:
        chunks = []
        for start_idx in range(0, total_frames - WINDOW_SIZE + 1, STRIDE):
            chunks.append(all_nodes[start_idx : start_idx + WINDOW_SIZE])
        
        if (total_frames - WINDOW_SIZE) % STRIDE != 0:
            chunks.append(all_nodes[-WINDOW_SIZE:])

    for chunk in chunks:
        input_tensor = format_tensor_chunk(chunk).to(WORD_DEVICE)
        
        with torch.no_grad():
            output = word_model(input_tensor)
            probs = torch.softmax(output, dim=1)
            conf, pred_idx = torch.max(probs, dim=1)
            
            conf_val = conf.item()
            pred_class = pred_idx.item()
            predicted_word = label_map.get(pred_class, "Unknown")
            
            if conf_val >= CONFIDENCE_THRESHOLD:
                raw_predictions.append({
                    "label": predicted_word,
                    "confidence": conf_val
                })
            
    # Post-Processing: Collapse consecutive identical words
    final_sentence = []
    for item in raw_predictions:
        if not final_sentence or final_sentence[-1]["label"] != item["label"]:
            final_sentence.append(item)

    if not final_sentence:
        return None, "No confident signs detected."

    return final_sentence, None

# --- REST OF APP CONFIGURATION ---

def generate_sentence(words):
    global LAST_GEMINI_ERROR

    prompt = f"""
You are a sign language translator.

Convert the sign language gloss words into a simple natural English sentence.

STRICT RULES:

* Do NOT replace words with synonyms.
* Use the SAME words given in the input.
* Only fix grammar.
* Only add helper words like "to", "am", "is", "are", "the", "a", "an", "do", "does", "did" if needed.
* Do NOT change verbs.
* Do NOT add new meaning.
* Keep all original words.
* Return ONLY the sentence.
* No quotes.
* No explanations.
* ADD SUBJECT to the sentence, ASSUME the subject "I" if not given in gross words

Make sure the sentence create the basic structure of English grammar and does not feel incomplete or broken but also DO NOT add new information 
IF ANY EXTRA WORD IS ADDED BY THE MODEL, AND IT IS NOT MATCHING THE REST OF THE SENTENCE THEN IT IS LIKELY THE MODEL IS HALLUCINATING, IN THAT CASE RETURN THE ORIGINAL SETENCE WITHOUT THE HALLUCINATED WORDS.
GRAMMAR GUIDELINES:

1. Pronouns
   ME → I
   YOU → you
   WE → we

2. Verb Structure
   If ME appears before a verb, convert to:
   ME VERB → I want to VERB

Examples:
ME EAT → I want to eat
ME DRINK → I want to drink
ME SLEEP → I want to sleep

3. Basic Word Order
   Try to follow:
   Subject → Verb → Object

Example:
YOU HELP ME → you help me

4. Politeness Words
   PLEASE, THANKYOU, SORRY stay in the sentence but can move to the start or end.

Example:
ME HELP PLEASE → I want to help please

5. Question Words
   WHAT, WHERE, WHY stay at the beginning.

Example:
WHAT YOU WANT → what you want
WHERE YOU GO → where you go

6. Do NOT change vocabulary words.

Examples:
EAT must stay "eat"
DRINK must stay "drink"
GO must stay "go"

EXAMPLES:

Sign: ME EAT
English: I want to eat

Sign: ME DRINK PLEASE
English: I want to drink please

Sign: YOU HELP ME
English: you help me

Sign: WE GO
English: we go

Sign: WHAT YOU WANT
English: what you want

Sign: WHERE YOU GO
English: where you go

Sign: ME NEED HELP
English: I need help

Sign: ME SLEEP
English: I want to sleep

Now translate:

Sign: {words}
English:
"""
    try:
        if not os.getenv("GOOGLE_API_KEY"):
            LAST_GEMINI_ERROR = "GOOGLE_API_KEY is not set in environment."
            print("LLM error:", LAST_GEMINI_ERROR)
            return words

        normalized_words = _normalize_cache_key(words)
        if normalized_words in GEMINI_CACHE:
            return GEMINI_CACHE[normalized_words]

        client = genai.Client()
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt
        )

        result = (response.text or "").strip()
        if not result:
            LAST_GEMINI_ERROR = "Gemini returned empty response."
            return words

        # remove explanations
        if ":" in result:
            result = result.split(":",1)[1].strip()

        result = result.split("\n")[0].strip()
        LAST_GEMINI_ERROR = None

        if normalized_words:
            GEMINI_CACHE[normalized_words] = result
            _save_json_cache(GEMINI_CACHE, GEMINI_CACHE_PATH)

        return result

    except Exception as e:
        LAST_GEMINI_ERROR = str(e)
        print("LLM error:", e)
        return words

def text_to_speech(text):

    if not text:
        return None

    audio_filename = f"speech_{int(time.time() * 1000)}.mp3"
    audio_path = os.path.join("static", audio_filename)

    try:
        tts = gTTS(text=text, lang="en")
        tts.save(audio_path)
        # Return web path so templates/JS can play it from any route.
        return f"/static/{audio_filename}"
    except Exception as e:
        print("TTS error:", e)
        return None


def translate_and_speak(words):
    sentence = generate_sentence(words)
    audio_path = text_to_speech(sentence) if sentence else None
    used_fallback = sentence.strip() == words.strip()
    return sentence, audio_path, used_fallback

def process_letter_number(video_path, mode):

    cap = cv2.VideoCapture(video_path)

    output_text = []

    with mp_hands.Hands(
        min_detection_confidence=0.9,
        min_tracking_confidence=0.9,
        max_num_hands=1
    ) as hands:

        while cap.isOpened():

            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = hands.process(rgb)

            if results.multi_hand_landmarks:

                for hand in results.multi_hand_landmarks:

                    x = []
                    y = []
                    data = []

                    for lm in hand.landmark:
                        x.append(lm.x)
                        y.append(lm.y)

                    for lm in hand.landmark:
                        data.append(lm.x - min(x))
                        data.append(lm.y - min(y))

                    if mode == "letter":
                        model = letter_model
                        threshold = LETTER_CONFIDENCE_THRESHOLD
                    else:
                        model = number_model
                        threshold = NUMBER_CONFIDENCE_THRESHOLD

                    sample = np.asarray(data)

                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba([sample])[0]
                        pred_idx = int(np.argmax(probs))
                        conf = float(probs[pred_idx])
                        if conf < threshold:
                            continue
                        if hasattr(model, "classes_"):
                            pred_label = model.classes_[pred_idx]
                        else:
                            pred_label = model.predict([sample])[0]
                    else:
                        pred_label = model.predict([sample])[0]

                    output_text.append(str(pred_label))

    cap.release()

    return " ".join(output_text)

def _normalize_mode(mode):
    if mode not in VALID_MODES:
        return "letter"
    return mode


def _get_live_state(mode):

    with LIVE_LOCK:

        state = LIVE_STATE[mode]

        return {
            "pending": state["pending"],
            "confirmed": list(COMBINED_OUTPUT)
        }


def _auto_accept_prediction(mode, prediction):
    if not prediction:
        return

    global COMBINED_OUTPUT
    now = time.time()

    with LIVE_LOCK:
        state = LIVE_STATE[mode]
        state["pending"] = prediction

        if state["streak_label"] == prediction:
            state["streak_count"] += 1
        else:
            state["streak_label"] = prediction
            state["streak_count"] = 1

        if state["streak_count"] < AUTO_ACCEPT_MIN_STREAK[mode]:
            return

        cooldown_ok = (
            prediction != state["last_added"] or
            (now - state["last_added_at"]) >= AUTO_ACCEPT_COOLDOWN_SEC[mode]
        )
        if not cooldown_ok:
            return

        state["confirmed"].append(prediction)
        COMBINED_OUTPUT.append(prediction)
        state["last_added"] = prediction
        state["last_added_at"] = now
        state["streak_count"] = 0


def _clear_live(mode):

    global COMBINED_OUTPUT

    with LIVE_LOCK:

        COMBINED_OUTPUT = []

        for m in LIVE_STATE:
            LIVE_STATE[m]["pending"] = None
            LIVE_STATE[m]["confirmed"] = []
            LIVE_STATE[m]["streak_label"] = None
            LIVE_STATE[m]["streak_count"] = 0
            LIVE_STATE[m]["last_added"] = None
            LIVE_STATE[m]["last_added_at"] = 0.0


def _predict_letter_number_frame(frame, hands, mode):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if not results.multi_hand_landmarks:
        return None

    hand = results.multi_hand_landmarks[0]

    x = []
    y = []
    data = []

    for lm in hand.landmark:
        x.append(lm.x)
        y.append(lm.y)

    for lm in hand.landmark:
        data.append(lm.x - min(x))
        data.append(lm.y - min(y))

    if mode == "letter":
        model = letter_model
        threshold = LETTER_CONFIDENCE_THRESHOLD
    else:
        model = number_model
        threshold = NUMBER_CONFIDENCE_THRESHOLD

    sample = np.asarray(data)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba([sample])[0]
        pred_idx = int(np.argmax(probs))
        conf = float(probs[pred_idx])
        if conf < threshold:
            return None
        if hasattr(model, "classes_"):
            pred_label = model.classes_[pred_idx]
        else:
            pred_label = model.predict([sample])[0]
    else:
        pred_label = model.predict([sample])[0]

    return str(pred_label)


def _iter_live_frames(mode):
    mode = _normalize_mode(mode)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return

    try:
        if mode in ("letter", "number"):
            with mp_hands.Hands(
                min_detection_confidence=0.9,
                min_tracking_confidence=0.9,
                max_num_hands=1
            ) as hands:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame = cv2.flip(frame, 1)

                    prediction = _predict_letter_number_frame(frame, hands, mode)
                    
                    if prediction:
                        _auto_accept_prediction(mode, prediction)
                            
                    with LIVE_LOCK:
                        pending = LIVE_STATE[mode]["pending"]

                    overlay_text = pending if pending else "Waiting..."
                    cv2.rectangle(frame, (10, 10), (500, 60), (0, 0, 0), -1)
                    cv2.putText(
                        frame,
                        f"Prediction: {overlay_text}",
                        (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA
                    )

                    ret, buffer = cv2.imencode(".jpg", frame)
                    if not ret:
                        continue
                    frame_bytes = buffer.tobytes()
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                    )
        else:
            MIN_FRAMES = 20  # Increased from 15 to reduce frequency
            FRAME_SKIP = 3  # Process every 3rd frame instead of 2nd
            frame_count = 0
            
            # Set camera to lower resolution for better performance
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 15)  # Limit to 15 FPS
            
            pose_mod_live = mp_pose.Pose(
                model_complexity=0,
                min_detection_confidence=0.6,  # Increased from 0.5
                min_tracking_confidence=0.6
            )
            hands_mod_live = mp_hands.Hands(
                max_num_hands=1,  # Reduced from 2 to 1 for speed
                min_detection_confidence=0.6,  # Increased from 0.5
                min_tracking_confidence=0.6
            )

            nodes_buffer = deque(maxlen=SEQ_LEN)
            
            # Performance monitoring
            import time
            last_inference_time = 0
            inference_count = 0

            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_count += 1
                    if frame_count % FRAME_SKIP != 0:
                        # Skip this frame for processing, but still display it
                        with LIVE_LOCK:
                            pending = LIVE_STATE[mode]["pending"]
                        overlay_text = f"{pending if pending else 'Waiting...'} | Skip"
                        cv2.rectangle(frame, (10, 10), (500, 60), (0, 0, 0), -1)
                        cv2.putText(
                            frame,
                            overlay_text,
                            (20, 45),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2,
                            cv2.LINE_AA
                        )
                        ret, buffer = cv2.imencode(".jpg", frame)
                        if ret:
                            frame_bytes = buffer.tobytes()
                            yield (
                                b"--frame\r\n"
                                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                            )
                        continue

                    frame = cv2.flip(frame, 1)
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    res_p = pose_mod_live.process(rgb_frame)
                    res_h = hands_mod_live.process(rgb_frame)

                    p = np.zeros((33, 4), dtype=np.float32)
                    if res_p and res_p.pose_landmarks:
                        for i, lm in enumerate(res_p.pose_landmarks.landmark):
                            p[i] = [lm.x, lm.y, lm.z, getattr(lm, "visibility", 1.0)]

                    h = np.zeros((2, 21, 4), dtype=np.float32)
                    if res_h and res_h.multi_hand_landmarks:
                        for i, hland in enumerate(res_h.multi_hand_landmarks):
                            label = res_h.multi_handedness[i].classification[0].label
                            idx = 0 if label == "Left" else 1
                            for j, lm in enumerate(hland.landmark):
                                h[idx, j] = [lm.x, lm.y, lm.z, 1.0]

                    pose_sel = p[POSE_KEEP, :]
                    left = h[0, :, :]
                    right = h[1, :, :]
                    nodes = np.concatenate([pose_sel, left, right], axis=0) # Shape: (53, 4)
                    
                    nodes_buffer.append(nodes)

                    with LIVE_LOCK:
                        pending = LIVE_STATE[mode]["pending"]

                    if len(nodes_buffer) >= MIN_FRAMES and frame_count % 5 == 0:  # Only infer every 5th processed frame
                        start_time = time.time()
                        # Convert buffer into array of shape (Time, Nodes, Features)
                        nodes_chunk = np.stack(list(nodes_buffer)) # (T, 53, 4)
                        
                        input_tensor = format_tensor_chunk(nodes_chunk).to(WORD_DEVICE)

                        with torch.no_grad():
                            logits = word_model(input_tensor)
                            probs = torch.softmax(logits, dim=1)
                            conf, pred_idx = probs.max(dim=1)

                        inference_time = time.time() - start_time
                        last_inference_time = inference_time
                        inference_count += 1
                        
                        conf_value = float(conf.item())
                        pred_idx_value = int(pred_idx.item())

                        if conf_value > CONFIDENCE_THRESHOLD:
                            prediction = label_map.get(pred_idx_value, "Unknown")
                            _auto_accept_prediction(mode, prediction)

                    with LIVE_LOCK:
                        pending = LIVE_STATE[mode]["pending"]

                    overlay_text = f"{pending if pending else 'Waiting...'} | Infer: {last_inference_time:.2f}s"
                    cv2.rectangle(frame, (10, 10), (600, 60), (0, 0, 0), -1)
                    cv2.putText(
                        frame,
                        overlay_text,
                        (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA
                    )

                    ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])  # Lower quality for faster encoding
                    if not ret:
                        continue
                    frame_bytes = buffer.tobytes()
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                    )
            finally:
                try:
                    pose_mod_live.close()
                except Exception:
                    pass
                try:
                    hands_mod_live.close()
                except Exception:
                    pass
    finally:
        cap.release()

@app.route("/", methods=["GET", "POST"])
def index():

    result_text = None
    result_list = None
    audio_path = None

    if request.method == "POST":

        video = request.files["video"]
        mode = request.form["mode"]

        video_path, video_hash, video_web_path = save_uploaded_video(video)
        cache_key = _video_cache_key(video_hash, mode)
        cached_result = VIDEO_CACHE.get(cache_key)

        result_list = None
        err_text = None
        llm_sentence = None
        llm_audio = None

        if cached_result:
            result_list = cached_result.get("result_list")
            result_text = cached_result.get("result_text") or cached_result.get("error_text")
            llm_sentence = cached_result.get("llm_sentence")
            llm_audio = cached_result.get("llm_audio_path")
        else:
            if mode == "word":
                result_list, err_text = process_word(video_path)
                if result_list:
                    result_text = " ".join([item["label"] for item in result_list])
                else:
                    result_text = err_text
            else:
                words = process_letter_number(video_path, mode)
                if words:
                    result_text = words
                else:
                    result_text = None

            VIDEO_CACHE[cache_key] = {
                "result_text": result_text,
                "result_list": result_list,
                "error_text": err_text,
                "llm_sentence": None,
                "llm_audio_path": None
            }
            _save_json_cache(VIDEO_CACHE, VIDEO_CACHE_PATH)

        if result_text and not llm_sentence:
            normalized_result = _normalize_cache_key(result_text)
            if normalized_result in GEMINI_CACHE:
                llm_sentence = GEMINI_CACHE[normalized_result]
                if not llm_audio:
                    llm_audio = text_to_speech(llm_sentence)
                VIDEO_CACHE[cache_key]["llm_sentence"] = llm_sentence
                VIDEO_CACHE[cache_key]["llm_audio_path"] = llm_audio
                _save_json_cache(VIDEO_CACHE, VIDEO_CACHE_PATH)

        if not result_text:
            audio_path = None
        else:
            audio_path = llm_audio

        return render_template(
        "upload.html",
        video_path=video_web_path,
        result_text=result_text,
        result_list=result_list,
        audio_path=audio_path,
        llm_output=llm_sentence,
        video_hash=video_hash
        )
    return render_template("index.html")

@app.route("/live", methods=["GET"])
def live():
    mode = _normalize_mode(request.args.get("mode", "letter"))
    return render_template("live.html", mode=mode)


@app.route("/video_feed")
def video_feed():
    mode = _normalize_mode(request.args.get("mode", "letter"))
    return Response(
        _iter_live_frames(mode),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/live_state")
def live_state():
    mode = _normalize_mode(request.args.get("mode", "letter"))
    return jsonify(_get_live_state(mode))

@app.route("/confirm", methods=["POST"])
def confirm():
    # Kept for backward compatibility; live mode now auto-accepts predictions.
    return jsonify({"ok": True, "message": "Auto-accept mode enabled"})

@app.route("/clear_live", methods=["POST"])
def clear_live():
    mode = request.form.get("mode")
    if request.is_json:
        payload = request.get_json(silent=True) or {}
        mode = payload.get("mode", mode)
    mode = _normalize_mode(mode or "letter")
    _clear_live(mode)
    return jsonify({"ok": True})

@app.route("/translate_live", methods=["POST"])
def translate_live():

    mode = request.form.get("mode")

    if request.is_json:
        payload = request.get_json(silent=True) or {}
        mode = payload.get("mode", mode)

    mode = _normalize_mode(mode or "letter")

    with LIVE_LOCK:
        confirmed = list(COMBINED_OUTPUT)

    if not confirmed:
        return jsonify({"sentence": ""})

    if mode == "letter":
        words = "".join(confirmed)   # combine letters into word
    else:
        words = " ".join(confirmed)  # keep spaces for word mode

    sentence, audio_path, used_fallback = translate_and_speak(words)

    return jsonify({
        "words": words,
        "sentence": sentence,
        "audio": audio_path,
        "used_fallback": used_fallback,
        "gemini_error": LAST_GEMINI_ERROR
    })

@app.route("/translate_text", methods=["POST"])
def translate_text():
    words = request.form.get("words", "")
    if request.is_json:
        payload = request.get_json(silent=True) or {}
        words = payload.get("words", words)

    words = (words or "").strip()
    if not words:
        return jsonify({"error": "No words provided"}), 400

    sentence, audio_path, used_fallback = translate_and_speak(words)

    video_hash = request.form.get("video_hash")
    if request.is_json:
        payload = request.get_json(silent=True) or {}
        video_hash = payload.get("video_hash", video_hash)

    if video_hash and sentence:
        for key, item in VIDEO_CACHE.items():
            if key.startswith(f"{video_hash}:") and item.get("result_text") == words:
                item["llm_sentence"] = sentence
                item["llm_audio_path"] = audio_path
        _save_json_cache(VIDEO_CACHE, VIDEO_CACHE_PATH)

    return jsonify({
        "words": words,
        "sentence": sentence or "",
        "audio": audio_path,
        "used_fallback": used_fallback,
        "gemini_error": LAST_GEMINI_ERROR
    })

if __name__ == "__main__":
    app.run(debug=True)