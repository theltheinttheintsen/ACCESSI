from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
from anthropic import Anthropic
import os, requests, json, tempfile, math
from dotenv import load_dotenv

load_dotenv(override=True)

print("Anthropic key starts with:", repr(os.getenv("ANTHROPIC_API_KEY", "")[:15]))
print("ORS key starts with:", repr(os.getenv("ORS_API_KEY", "")[:8]))
print("Deepgram key starts with:", repr(os.getenv("DEEPGRAM_API_KEY", "")[:8]))
print("ORS key starts with:", repr(os.getenv("ORS_API_KEY", "")[:10]))

app = Flask(__name__)
CORS(app)

claude_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
CLAUDE_MODEL = "claude-haiku-4-5-20251001"  # fast + cheap, good fit for short narrations

ORS_KEY = os.getenv("ORS_API_KEY")
DEEPGRAM_KEY = os.getenv("DEEPGRAM_API_KEY")

# Video calibration deps are optional — the app still runs fine without them,
# /calibrate-video just returns a clear error telling you to pip install them.
try:
    import cv2
    import numpy as np
    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision
    VIDEO_CALIBRATION_AVAILABLE = True
except ImportError:
    VIDEO_CALIBRATION_AVAILABLE = False

POSE_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pose_landmarker_lite.task')
POSE_MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task'


def ensure_pose_model():
    """Download the pose-detection model once, on first use. Subsequent
    calls reuse the saved file. ~5-6MB, one-time download."""
    if not os.path.exists(POSE_MODEL_PATH):
        print("Downloading pose landmark model (one-time, ~5-6MB)...")
        r = requests.get(POSE_MODEL_URL, timeout=60)
        r.raise_for_status()
        with open(POSE_MODEL_PATH, 'wb') as f:
            f.write(r.content)
        print("Pose model downloaded to", POSE_MODEL_PATH)

ORS_PROFILES = {
    "wheelchair": "wheelchair",
    "blind":      "foot-walking",
    "elderly":    "foot-walking",
    "disabled":   "wheelchair",
    "stroller":   "wheelchair",
}


def reverse_geocode(lat, lng):
    """Convert raw coordinates into a human-readable place name."""
    try:
        res = requests.get(
            "https://api.openrouteservice.org/geocode/reverse",
            params={"api_key": ORS_KEY, "point.lon": lng, "point.lat": lat, "size": 1},
            timeout=5
        )
        data = res.json()
        label = data["features"][0]["properties"].get("label")
        return label or "your current location"
    except Exception:
        return "your current location"


def haversine_meters(lat1, lng1, lat2, lng2):
    """Great-circle distance between two points, in meters."""
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lng2 - lng1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def geocode_destination(query, focus_lat=None, focus_lng=None):
    """Convert a typed destination into coordinates.

    For chains/franchises ('Safeway', '99 Ranch', 'HeyTea') that have many
    locations, ORS's relevance ranking alone can return the wrong branch.
    To fix this properly:
      1. We bias toward the user's area with focus.point — but as a soft
         hint only, NOT a hard radius cutoff. A hard radius would wrongly
         exclude the correct branch if the person explicitly names a
         different city (e.g. '99 Ranch Fremont' while testing from
         somewhere far from Fremont) — exactly the case this is meant to fix.
      2. Pelias scores every result with a 'confidence' (0-1) measuring how
         well it actually matches the text query. We only compare DISTANCE
         among results that are genuinely good text matches — this stops an
         irrelevant-but-nearby place from beating the real, correctly-named,
         slightly farther store."""
    params = {"api_key": ORS_KEY, "text": query, "size": 10 if focus_lat is not None else 1}
    if focus_lat is not None and focus_lng is not None:
        params["focus.point.lon"] = focus_lng
        params["focus.point.lat"] = focus_lat
    try:
        res = requests.get(
            "https://api.openrouteservice.org/geocode/search",
            params=params,
            timeout=5
        )
        data = res.json()
        features = data.get("features", [])
        if not features:
            print("ORS GEOCODE: no features returned. Status:", res.status_code, "Response:", data)
            return None

        if focus_lat is not None and focus_lng is not None and len(features) > 1:
            top_confidence = max(f["properties"].get("confidence", 0) for f in features)
            # Only consider results that are close in quality to the best
            # text match — never let a low-relevance result win purely on
            # proximity.
            relevant = [
                f for f in features
                if f["properties"].get("confidence", 0) >= max(top_confidence - 0.15, 0.4)
            ]
            if not relevant:
                relevant = [features[0]]

            best = min(
                relevant,
                key=lambda f: haversine_meters(
                    focus_lat, focus_lng,
                    f["geometry"]["coordinates"][1], f["geometry"]["coordinates"][0]
                )
            )
        else:
            best = features[0]

        lng, lat = best["geometry"]["coordinates"]
        label = best["properties"].get("label", query)
        return {"lat": lat, "lng": lng, "label": label}
    except Exception as e:
        print("ORS GEOCODE ERROR:", e)
        return None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/route', methods=['POST'])
def get_route():
    """Returns real GeoJSON route + plain-English step list."""
    data = request.json
    start = data.get('start')
    destination_text = data.get('destination')
    user_type = data.get('userType', 'wheelchair')

    if not start or not destination_text:
        return jsonify({"error": "start location and destination required"}), 400

    dest = geocode_destination(destination_text, start.get('lat'), start.get('lng'))
    if not dest:
        return jsonify({"error": f"Could not find '{destination_text}'. Try a more specific address."}), 404

    profile = ORS_PROFILES.get(user_type, "wheelchair")
    url = f"https://api.openrouteservice.org/v2/directions/{profile}/geojson"

    try:
        res = requests.post(url,
            headers={"Authorization": ORS_KEY, "Content-Type": "application/json"},
            json={
                "coordinates": [[start['lng'], start['lat']], [dest['lng'], dest['lat']]],
                "instructions": True,
                "language": "en",
                "units": "m",
            },
            timeout=10
        )
        route_data = res.json()

        if "error" in route_data:
            return jsonify({"error": route_data["error"]["message"]}), 500

        summary  = route_data["features"][0]["properties"]["summary"]
        segments = route_data["features"][0]["properties"]["segments"]
        coords   = route_data["features"][0]["geometry"]["coordinates"]

        steps = []
        for seg in segments:
            for step in seg["steps"]:
                steps.append({
                    "instruction": step["instruction"],
                    "distance_m":  round(step["distance"]),
                    "duration_s":  round(step["duration"]),
                })

        return jsonify({
            "destination_label": dest["label"],
            "destination_lat": dest["lat"],
            "destination_lng": dest["lng"],
            "route_coords": coords,
            "steps": steps,
            "distance_m": round(summary["distance"]),
            "duration_s": round(summary["duration"]),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/ask', methods=['POST'])
def ask():
    """AI narrator — turns route steps into a warm, human-friendly summary.
    Never receives or echoes raw lat/lng."""
    data = request.json
    user_type = data.get('userType', 'wheelchair')
    destination_label = data.get('destinationLabel', 'your destination')
    steps = data.get('steps', [])
    distance_m = data.get('distanceM')
    duration_s = data.get('durationS')
    stride_m = data.get('strideM')  

    steps_text = "\n".join(
        f"- {s['instruction']} ({s['distance_m']}m)" for s in steps
    ) or "No detailed steps available."

    if stride_m:
        stride_instruction = f'For blind users: convert metres to walking steps using THIS user\'s calibrated step length of {stride_m:.2f}m per step (measured from their own walking pace) and emphasise step counts.'
    else:
        stride_instruction = 'For blind users: convert metres to walking steps (1 step \u2248 0.75m, a general estimate) and emphasise step counts.'

    system_prompt = f"""You are AccessPath's AI navigation narrator for disabled people.
You will be given a list of real turn-by-turn directions. Your job is to rewrite them
as a short, warm, easy-to-follow spoken summary.

STRICT RULES:
- NEVER mention coordinates, latitude, longitude, or any raw numbers like "37.8724".
- Only reference street names, distances in metres or steps, and landmarks already given.
- EVERY turn or instruction you mention MUST include how far to go before or after it — never say "turn right, then turn left" without a distance for each leg. Even very short distances must be stated (e.g. "in just 10 meters, turn left" or "after a few steps, turn right"). Do not skip distances for brevity.
- {stride_instruction}
- For wheelchair users: mention ramps, curb cuts, and flag any stairs as a warning.
- For elderly users: suggest pacing and mention if the route is long enough to need a rest.
- Keep it under 80 words. Be warm and encouraging. End with arrival confirmation.
- If a route has many turns and space is tight, shorten the warmth/flourish first — distances are mandatory safety information and must NEVER be the thing you cut.
"""

    user_msg = f"""User type: {user_type}
Destination: {destination_label}
Total distance: {distance_m}m
Total time: {duration_s}s

Turn-by-turn steps:
{steps_text}

Rewrite this as a short, friendly spoken-style summary for the user."""

    try:
        response = claude_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=400,
            system=system_prompt,
            messages=[{"role": "user", "content": user_msg}]
        )
        text = response.content[0].text
        return jsonify({"response": text})
    except Exception as e:
        print("CLAUDE ERROR:", e)
        return jsonify({"response": "AI narration is temporarily unavailable, but your route is ready below.", "error": str(e)}), 200


@app.route('/safety-score', methods=['POST'])
def safety_score():
    """AI-ESTIMATED accessibility score for a route, based on its shape
    (turn count, distance, instructions). This is a heuristic estimate from
    Claude reasoning about the route data we already have — not measured
    real-world sidewalk/curb data. Always presented to the user as an estimate."""
    data = request.json or {}
    user_type = data.get('userType', 'wheelchair')
    steps = data.get('steps', [])
    distance_m = data.get('distanceM')
    duration_s = data.get('durationS')

    steps_text = "\n".join(
        f"- {s['instruction']} ({s['distance_m']}m)" for s in steps
    ) or "No detailed steps available."

    system_prompt = """You are an accessibility scoring assistant for a navigation app.
Given a route's turn-by-turn instructions, distance, and the traveler's needs, estimate
a rough accessibility score for THIS route.

You do NOT have real-time sidewalk/curb condition data. Reason only from what the
route shape suggests (number of turns/crossings, total distance and time, route
type) — do not invent specific facts like exact curb cuts or named obstacles.

Respond with ONLY valid JSON, no other text, in exactly this shape:
{"score": <integer 0-100>, "label": "<2-4 word label>", "reasons": ["<reason 1>", "<reason 2>", "<reason 3>"]}

Score guide:
90-100 = very accessible: short, few turns
70-89  = mostly accessible: minor concerns
40-69  = moderate concerns: several turns/crossings or longer distance
0-39   = significant concerns: long and complex route

Each reason under 12 words, specific to this route's shape (turn count, distance, duration)."""

    user_msg = f"""Traveler type: {user_type}
Total distance: {distance_m}m
Total time: {duration_s}s
Number of turns/steps: {len(steps)}

Turn-by-turn steps:
{steps_text}

Return the JSON accessibility estimate now."""

    try:
        response = claude_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=300,
            system=system_prompt,
            messages=[{"role": "user", "content": user_msg}]
        )
        text = response.content[0].text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
        parsed = json.loads(text)
        score = max(0, min(100, int(parsed.get("score", 70))))
        return jsonify({
            "score": score,
            "label": parsed.get("label", "Estimated"),
            "reasons": parsed.get("reasons", [])[:3]
        })
    except Exception as e:
        print("SAFETY SCORE ERROR:", e)
        return jsonify({"score": None, "label": "Unavailable", "reasons": []}), 200


@app.route('/report', methods=['POST'])
def report():
    data = request.json
    obstacle_type = data.get('type', '')
    location = data.get('location', {})

    place = reverse_geocode(location.get('lat'), location.get('lng')) if location else "an unknown location"

    try:
        response = claude_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=100,
            system="You are AccessPath's obstacle reporting assistant. Acknowledge warmly and give a brief safety tip. Never mention coordinates. Max 2 sentences.",
            messages=[{"role": "user", "content": f"Reported obstacle: {obstacle_type} near {place}"}]
        )
        message = response.content[0].text
    except Exception as e:
        print("CLAUDE ERROR (report):", e)
        message = f"Thanks for reporting the {obstacle_type.lower()}! Stay safe out there."

    return jsonify({
        "message": message,
        "obstacle": obstacle_type,
        "location_label": place
    })


@app.route('/calibrate-video', methods=['POST'])
def calibrate_video():
    """Estimate the user's real average step length (blind/elderly) or
    real wheelchair speed (wheelchair/disabled/stroller) from a short
    walking/rolling video, using their height as a real-world scale
    reference.

    IMPORTANT HONESTY NOTE: this is an ESTIMATE. Monocular video gait
    analysis has real error margins (camera angle, lighting, clothing,
    and walking consistency all affect accuracy). It is not a medical
    or precision measurement tool."""
    if not VIDEO_CALIBRATION_AVAILABLE:
        return jsonify({
            "error": "Video calibration isn't set up on this server yet. "
                     "Run: pip install opencv-python-headless mediapipe numpy"
        }), 500

    if 'video' not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    height_cm = request.form.get('heightCm', type=float)
    user_type = request.form.get('userType', 'blind')

    if not height_cm or height_cm < 50 or height_cm > 250:
        return jsonify({"error": "Please enter a valid height in centimetres."}), 400

    try:
        ensure_pose_model()
    except Exception as e:
        print("POSE MODEL DOWNLOAD ERROR:", e)
        return jsonify({"error": "Could not download the pose-detection model. Check your internet connection and try again."}), 500

    video_file = request.files['video']
    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            video_file.save(tmp.name)
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            return jsonify({"error": "Could not read the uploaded video file."}), 400

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        base_options = mp_tasks.BaseOptions(model_asset_path=POSE_MODEL_PATH)
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
        )
        landmarker = mp_vision.PoseLandmarker.create_from_options(options)

        is_wheelchair_type = user_type in ('wheelchair', 'disabled', 'stroller')

        separations = []      
        heights_px = []       
        hip_positions = []    

        frame_idx = 0
        SAMPLE_EVERY = 2  # process every other frame to keep this reasonably fast

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % SAMPLE_EVERY != 0:
                continue

            timestamp_ms = int((frame_idx / fps) * 1000)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks:
                lm = result.pose_landmarks[0]  # first detected person
                h, w = frame.shape[:2]

                # BlazePose 33-point landmark indices:
                # 0 = nose, 23/24 = left/right hip, 27/28 = left/right ankle
                nose       = lm[0]
                left_hip   = lm[23]
                right_hip  = lm[24]
                left_ankle = lm[27]
                right_ankle = lm[28]

                nx, ny = nose.x * w, nose.y * h
                lx, ly = left_ankle.x * w, left_ankle.y * h
                rx, ry = right_ankle.x * w, right_ankle.y * h
                hip_x = ((left_hip.x + right_hip.x) / 2) * w

                separation_px = abs(lx - rx)
                avg_ankle_y = (ly + ry) / 2
                height_px = abs(avg_ankle_y - ny)

                if height_px > 10:  # sanity guard against bad detections
                    separations.append(separation_px)
                    heights_px.append(height_px)
                    hip_positions.append((timestamp_ms, hip_x))

        cap.release()
        landmarker.close()

        if len(separations) < 10:
            return jsonify({"error": "Could not clearly detect a person walking. Make sure your full body is visible from the side, with good lighting, and try again."}), 200

        median_height_px = float(np.median(heights_px))
        meters_per_px = (height_cm / 100.0) / (median_height_px / 0.87)

        if is_wheelchair_type:
            if len(hip_positions) < 10:
                return jsonify({"error": "Not enough motion detected. Try rolling further across the frame."}), 200

            t0, x0 = hip_positions[0]
            t1, x1 = hip_positions[-1]
            elapsed_s = (t1 - t0) / 1000.0
            distance_px = abs(x1 - x0)
            distance_m = distance_px * meters_per_px

            if elapsed_s < 1:
                return jsonify({"error": "Video is too short to measure speed. Try a longer clip."}), 200

            mps = distance_m / elapsed_s
            mph = mps * 2.23694
            mph = max(0.3, min(8.0, mph))  # sanity clamp to plausible wheelchair speeds

            return jsonify({
                "wheelchair_mph": round(mph, 2),
                "frames_analyzed": len(hip_positions)
            })

        else:
            avg_sep = sum(separations) / len(separations)
            peak_separations = []
            for i in range(1, len(separations) - 1):
                if separations[i] > separations[i - 1] and separations[i] > separations[i + 1]:
                    if separations[i] > avg_sep * 1.1:
                        peak_separations.append(separations[i])

            if len(peak_separations) < 3:
                return jsonify({"error": "Not enough clear steps detected. Try walking more steps, fully in frame, from the side."}), 200

            avg_separation_px = sum(peak_separations) / len(peak_separations)
            stride_m = avg_separation_px * meters_per_px
            stride_m = max(0.35, min(1.2, stride_m))  # sanity clamp to plausible human strides

            return jsonify({
                "stride_m": round(stride_m, 3),
                "steps_detected": len(peak_separations),
                "frames_analyzed": len(separations)
            })

    except Exception as e:
        print("VIDEO CALIBRATION ERROR:", e)
        return jsonify({"error": f"Video processing failed: {str(e)}"}), 500
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


@app.route('/transcribe', methods=['POST'])
def transcribe():
    """Speech-to-text via Deepgram. Expects raw audio bytes in the request body,
    with the recorded mimetype passed through as the Content-Type header."""
    audio_bytes = request.get_data()
    mimetype = request.content_type or "audio/webm"

    if not audio_bytes:
        return jsonify({"error": "No audio received"}), 400

    try:
        res = requests.post(
            "https://api.deepgram.com/v1/listen",
            headers={
                "Authorization": f"Token {DEEPGRAM_KEY}",
                "Content-Type": mimetype,
            },
            params={
                "model": "nova-2",
                "smart_format": "true",
                "punctuate": "true",
                "language": "en-US",
            },
            data=audio_bytes,
            timeout=15
        )
        data = res.json()
        transcript = (
            data.get("results", {})
                .get("channels", [{}])[0]
                .get("alternatives", [{}])[0]
                .get("transcript", "")
        )
        if not transcript:
            print("DEEPGRAM STT RESPONSE:", data)
            return jsonify({"error": "Could not understand audio"}), 200
        return jsonify({"transcript": transcript})
    except Exception as e:
        print("DEEPGRAM STT ERROR:", e)
        return jsonify({"error": str(e)}), 500


@app.route('/speak', methods=['POST'])
def speak():
    """Text-to-speech via Deepgram Aura. Returns raw MP3 bytes."""
    data = request.json or {}
    text = data.get('text', '').strip()

    if not text:
        return jsonify({"error": "text required"}), 400

    try:
        res = requests.post(
            "https://api.deepgram.com/v1/speak",
            headers={
                "Authorization": f"Token {DEEPGRAM_KEY}",
                "Content-Type": "application/json",
            },
            params={"model": "aura-2-thalia-en"},
            json={"text": text},
            timeout=20
        )
        if res.status_code != 200:
            print("DEEPGRAM TTS ERROR:", res.text)
            return jsonify({"error": f"Deepgram TTS error: {res.text}"}), 500

        return Response(res.content, mimetype="audio/mpeg")
    except Exception as e:
        print("DEEPGRAM TTS ERROR:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)