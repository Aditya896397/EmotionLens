# Interview preparation: EmotionLens AI

Answers are short on purpose: say them in your own words and point to the file that implements each one.
Numbers quoted below come from the measured results in `reports/experiments/` (see README → Results).

## Project pitch (30 seconds)
"EmotionLens AI is a real-time computer vision system that detects multiple faces, tracks them with
ByteTrack, classifies their visible facial expressions with a CNN trained on FER-2013, and smooths
predictions over time per person. A Streamlit dashboard shows the analytics, and only aggregate
statistics are stored in SQLite. I compared a custom CNN with MobileNetV2 transfer learning:
64.0% vs 55.0% test accuracy (macro-F1 0.619 vs 0.501) at 2.6 vs 1.0 ms per face on a laptop CPU."

## Architecture and design
1. **Walk me through one frame.** Brightness check → (enhance) → BlazeFace → ByteTrack update → align +
   crop each face to 48×48 → one batched CNN call → per-track EMA + hysteresis → session stats → overlay.
   (`recognition/pipeline.py`)
2. **Why put normalisation/augmentation inside the model?** The model carries its own preprocessing, so
   there is no training/serving skew, models are swappable behind one contract, and augmentation layers are
   inactive at inference automatically.
3. **Why Streamlit, and what's hard about it for video?** Fast to build dashboards. But it reruns the whole
   script on each interaction, so I cache the model (`st.cache_resource`), keep live state in
   `session_state`, and release the camera with a context manager when the rerun interrupts the loop.
4. **How would you scale this to many cameras?** Separate capture/inference into worker processes or
   a service (e.g., FastAPI + queue), batch faces across streams on a GPU, export to TFLite/ONNX
   Runtime/OpenVINO, and push events to a time-series store instead of SQLite.

## Detection, alignment and tracking
5. **Why MediaPipe instead of YOLO?** It runs in real time on a CPU, gives eye keypoints for alignment,
   and installs as one pip package. YOLO-face would need PyTorch and a GPU to be comfortable.
6. **Detection confidence vs expression confidence?** The first is the detector's probability that a box is
   a face; the second is the classifier's softmax probability for a class. They are independent, and
   softmax confidence is not calibrated accuracy.
7. **How does ByteTrack work?** Kalman predict → match high-score detections to all tracks by IoU with the
   Hungarian algorithm → match low-score detections to remaining active tracks → age out lost tracks
   → start tentative tracks. (`detection/face_tracker.py`)
8. **Why use low-score detections at all?** Occlusion and motion blur lower detector scores. Discarding them
   breaks tracks; matching them only to *existing* tracks keeps continuity without spawning false tracks.
9. **What does the Kalman filter add over plain IoU matching?** It predicts where a moving face will be, so
   IoU is computed against the predicted box. That keeps fast motion and brief occlusions matchable.
10. **Why not DeepSORT?** It needs appearance embeddings, which are biometric descriptors, plus an extra model
    and latency. The trade-off is weaker re-identification after long absences or crossings.
11. **What is the Hungarian algorithm's complexity?** O(n³). That's fine for a handful of faces per frame.
12. **How does alignment work?** The eye-line angle is computed with `atan2`, then one affine warp rotates,
    scales and crops in one step, rendering only 48×48 pixels.

## Model and training
13. **Describe your CNN.** 4 blocks of double Conv-BN-ReLU + MaxPool + Dropout, 64→256 channels,
    Dense-256, softmax-7, about 2.9M parameters.
14. **Why BatchNorm and Dropout?** BN stabilises and speeds up training (it normalises activations and
    allows higher learning rates). Dropout and L2 reduce overfitting on a small, noisy dataset.
15. **How did you handle class imbalance?** Balanced class weights `N/(K·n_c)` in the loss, plus reporting
    macro-F1 and per-class recall. Alternatives: oversampling, focal loss, mixup.
16. **Why fine-tune MobileNetV2 in two stages?** Training a random head on a frozen backbone first avoids
    large gradients destroying the pretrained features. Then the top layers are unfrozen with a 10×
    lower learning rate.
17. **Why keep BatchNorm frozen during fine-tuning?** FER grayscale faces have very different statistics
    from ImageNet, and updating BN running stats with small batches destabilises the network.
18. **Which callbacks and why?** EarlyStopping (stop when val_loss stalls, restore the best weights),
    ReduceLROnPlateau (smaller steps near a minimum), ModelCheckpoint (keep the best val_accuracy model).
19. **Why upscale 48→96 for MobileNetV2?** ImageNet features are learned at higher resolution, and at
    48 px the stride-32 backbone would end at 2×2 feature maps.
20. **What's the accuracy ceiling on FER-2013?** Human accuracy is often quoted around 65±5%, and
    state-of-the-art single models are in the low-to-mid 70s. Label noise limits it.
21. **How did you avoid data leakage?** I hashed every image and removed training images identical to test
    images, plus within-train duplicates and blank images. The test split is left untouched.

## Evaluation and experiments
22. **Which metrics, and why macro-F1?** Accuracy is dominated by frequent classes; macro-F1 weighs every
    class equally, so a failure on *Disgust* shows up.
23. **What did the confusion matrix show?** Fear is the hardest class (recall 0.32–0.44), confused with Surprise and Sad; Sad is confused with Neutral. Happy and Surprise are easiest (recall ≈ 0.8).
24. **How did you measure latency?** `tf.function`, 10 warm-up runs, then median/p95 of 100 runs for batch
    sizes 1/4/8 on CPU. Batching makes the per-face cost drop as the face count grows.
25. **What did temporal smoothing do?** On synthetic jitter sequences, flips per 40 frames fell from 8.86 (raw)
    to 0.75 (EMA α = 0.35 + hysteresis), −92%, with frame accuracy unchanged (0.495 → 0.504). The cost is response lag to real changes.
26. **What about lighting robustness?** `cnn_noaug` drops from 0.640 to 0.363 under "very dark"; the pipeline's
    enhancement recovers it to 0.619. Augmentation alone keeps the CNN at 0.494. Synthetic darkening underestimates real low-light noise.
27. **What would you add next?** Real webcam test clips with labels, calibration (temperature scaling)
    so confidence ≈ accuracy, a demographic fairness audit on a dataset with annotations
    (e.g., RAF-DB/AffectNet subsets), and quantised TFLite inference.

## Engineering, privacy and ethics
28. **How do you guarantee the webcam is released?** `Camera` is a context manager, so `release()` runs in
    `__exit__` even when Streamlit interrupts the script or an exception is raised.
29. **What runs without a GPU?** Everything. TensorFlow detects devices once; on native Windows TF ≥ 2.11
    is CPU-only, which is why the models are kept small and inference is batched.
30. **What do you store?** Session aggregates, per-second labels per anonymous track ID, and track
    summaries. No pixels, no embeddings; one-click deletion; cascading deletes.
31. **Ethical risks?** Expression ≠ emotion, demographic bias, and surveillance or misuse in hiring,
    education and policing. This is why the disclaimers exist, nothing biometric is stored, and the
    system is not meant for decisions about individuals.
32. **Why did you pin library versions?** MediaPipe 0.10.21 needs numpy<2 and protobuf<5; TF 2.17.1 is
    compatible with both; mediapipe pulls opencv-contrib-python, which conflicts with opencv-python.

33. **Why ONNX Runtime on the laptop?** Smart App Control on Windows 11 blocked TensorFlow's unsigned DLLs.
    I split training (Colab GPU, TensorFlow) from serving (ONNX Runtime: Microsoft-signed, small, fast
    CPU kernels), verified the export numerically (max probability difference and label agreement vs
    Keras), and kept one classifier interface with two backends.
34. **What was the DLL load-order bug?** pyarrow bundles an older `msvcp140.dll`. If pandas imports pyarrow
    first, Windows reuses that DLL and MediaPipe's native module fails to initialise. Loading MediaPipe
    first (`utils/native_preload.py`) fixes it.

## Quick formulas
- IoU = |A∩B| / |A∪B|
- EMA: s_t = α·p_t + (1−α)·s_{t−1}; effective window ≈ 2/α − 1
- Balanced class weight: w_c = N / (K · n_c)
- Precision = TP/(TP+FP), Recall = TP/(TP+FN), F1 = 2PR/(P+R)
- Kalman: predict x̂ = Fx, P = FPFᵀ+Q; update K = PHᵀ(HPHᵀ+R)⁻¹, x = x̂ + K(z − Hx̂)
