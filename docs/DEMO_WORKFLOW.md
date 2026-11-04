# Demo workflow (about 8 minutes)

Prepare beforehand: models trained and evaluated, experiments run, one short MP4 with 2–3 people,
one group photo, good lighting plus a way to dim it (turn off a lamp).

1. **Home Dashboard (30 s)**: State the problem and the ethics note: *visible expression, not emotion*.
   Point out the system status (CPU device, active model, detector backend).
2. **Model Performance (1.5 min)**: Comparison table: custom CNN vs MobileNetV2 accuracy, macro-F1,
   parameters, training time, latency. Open a confusion matrix and point to the Fear↔Surprise and
   Sad↔Neutral confusions. Show per-class F1 next to the class imbalance chart (Disgust is rare).
3. **Live Webcam (2 min)**: Turn the camera on. Show the bounding box, track ID, label, expression
   confidence vs face detection confidence, FPS and latency KPIs.
   - Change expressions slowly: the label changes only after a few frames (hysteresis).
   - Cover your face briefly, then uncover it: the same ID comes back (lost-track re-identification).
   - Bring in a second person: a new ID appears and the face count updates.
   - Dim the light: the "low-light boost" tag appears in the HUD.
   Turn the camera off: the session is saved.
4. **Expression Analytics (1 min)**: Pick the session just recorded. Show the donut, frequency over time,
   per-track history and stability. Use the filters (track ID, time range). Download the PDF.
5. **Image Analysis (45 s)**: Upload the group photo: faces numbered left to right, top-3 probabilities,
   PDF/CSV/PNG export.
6. **Video Analysis (1 min)**: Upload the MP4 and analyse the first 20 s. Show the processed video,
   timeline and track history, then download the processed MP4.
7. **Session History and Settings (45 s)**: Delete a session (privacy). In Settings, show detector
   backend, thresholds, smoothing alpha and ByteTrack parameters, and explain one trade-off
   (higher α = faster reaction, more flicker).

Closing line: results come from measured experiments; limitations include FER-2013 bias, domain shift
to webcams, and that expressions are not emotions.
