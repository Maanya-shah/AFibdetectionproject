# ============================================================
# app.py — Flask Web Portal for AFib Detection
# ------------------------------------------------------------
# Run this with: python app.py
# Then open:     http://localhost:5000
# ============================================================

import os
import uuid
import numpy as np
import pandas as pd
import scipy.io
import matplotlib
matplotlib.use('Agg')   # Non-interactive backend — required for Flask (no display)
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from flask import Flask, request, render_template, jsonify
from scipy.signal import butter, filtfilt, iirnotch, find_peaks
import joblib
from tensorflow import keras
import tensorflow as tf

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs('uploads', exist_ok=True)
os.makedirs('static/results', exist_ok=True)

# ============================================================
# LOAD MODELS AT STARTUP (once, not on every request)
# ============================================================
print("Loading models...")
RF_MODEL   = joblib.load('models/random_forest.pkl')
SCALER     = joblib.load('models/scaler.pkl')
CNN_MODEL  = keras.models.load_model('models/cnn_lstm_best.keras')
CNN_THRESH = float(np.load('models/cnn_lstm_threshold.npy')[0])
FS         = 300
SEG_LEN    = 9000
print(f"Models loaded. CNN threshold = {CNN_THRESH:.3f}")

# ============================================================
# GRAD-CAM SETUP
# ============================================================
# Build a sub-model that outputs BOTH the last Conv1D feature maps
# AND the final probability — needed to compute gradients.
_last_conv = None
for _layer in CNN_MODEL.layers:
    if isinstance(_layer, keras.layers.Conv1D):
        _last_conv = _layer   # keep updating — last Conv1D wins

GRAD_MODEL = keras.Model(
    inputs  = CNN_MODEL.input,
    outputs = [_last_conv.output, CNN_MODEL.output]
)
print(f"Grad-CAM layer: {_last_conv.name}")


# ============================================================
# PREPROCESSING FUNCTIONS
# ============================================================

def preprocess_ecg(signal, fs=FS):
    """Bandpass + notch filter + z-score normalize."""
    nyq = fs / 2.0
    b, a = butter(4, [0.5/nyq, 40.0/nyq], btype='band')
    signal = filtfilt(b, a, signal)
    b, a = iirnotch(60.0/(fs/2.0), 30.0)
    signal = filtfilt(b, a, signal)
    return (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

def detect_r_peaks(signal, fs=FS):
    peaks, _ = find_peaks(signal, height=0.6*np.std(signal), distance=int(0.25*fs))
    return peaks

def extract_features(signal, fs=FS):
    """Extract the same 15 features used to train the traditional models."""
    peaks = detect_r_peaks(signal, fs)
    if len(peaks) < 3:
        return None
    rr       = np.diff(peaks) * (1000.0 / fs)
    diff_rr  = np.diff(rr)
    mean_rr  = np.mean(rr)
    std_rr   = np.std(rr)
    rmssd    = np.sqrt(np.mean(diff_rr**2))
    pnn50    = 100.0 * np.sum(np.abs(diff_rr) > 50) / len(diff_rr)
    cv_rr    = std_rr / (mean_rr + 1e-8)
    range_rr = np.max(rr) - np.min(rr)
    median_rr= np.median(rr)
    iqr_rr   = np.percentile(rr, 75) - np.percentile(rr, 25)
    rr1, rr2 = rr[:-1], rr[1:]
    sd1      = np.std((rr2 - rr1) / np.sqrt(2))
    sd2      = np.std((rr2 + rr1) / np.sqrt(2))
    sd_ratio = sd1 / (sd2 + 1e-8)
    counts, _ = np.histogram(rr, bins=10)
    probs    = counts / (counts.sum() + 1e-8)
    probs    = probs[probs > 0]
    entropy  = -np.sum(probs * np.log(probs))
    hr_bpm   = len(peaks) / (len(signal)/fs) * 60
    import scipy.stats
    skew     = scipy.stats.skew(signal)
    kurt     = scipy.stats.kurtosis(signal)
    return [mean_rr, std_rr, rmssd, pnn50, cv_rr, range_rr,
            median_rr, iqr_rr, entropy, sd1, sd2, sd_ratio,
            hr_bpm, skew, kurt]

def load_ecg_from_file(filepath):
    """Load ECG from .mat or .csv file. Returns raw signal array."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.mat':
        mat = scipy.io.loadmat(filepath)
        # Try common key names used by different devices
        for key in ['val', 'data', 'ecg', 'signal']:
            if key in mat:
                return mat[key].flatten().astype(float)
        # Fallback: use first non-metadata key
        data_keys = [k for k in mat.keys() if not k.startswith('_')]
        return mat[data_keys[0]].flatten().astype(float)
    elif ext == '.csv':
        df = pd.read_csv(filepath, header=None)
        # Use the first numeric column
        return df.iloc[:, 0].dropna().values.astype(float)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

# ============================================================
# CHA₂DS₂-VASc SCORE CALCULATOR
# ============================================================

# Annual stroke risk lookup table by score
STROKE_RISK = {0: 0.0, 1: 1.3, 2: 2.2, 3: 3.2, 4: 4.0,
               5: 6.7, 6: 9.8, 7: 9.6, 8: 12.5, 9: 15.2}

def calculate_chadsvasc(age, sex, hypertension, diabetes,
                         prior_stroke, heart_failure, vascular_disease):
    """
    Calculates CHA₂DS₂-VASc score from patient demographics and history.
    Returns (score, annual_stroke_risk_percent).
    """
    score = 0
    score += int(heart_failure)       # C: Congestive heart failure
    score += int(hypertension)        # H: Hypertension
    if age >= 75:
        score += 2                    # A₂: Age ≥ 75 = 2 points
    elif age >= 65:
        score += 1                    # A: Age 65–74 = 1 point
    score += int(diabetes)            # D: Diabetes
    score += 2 * int(prior_stroke)    # S₂: Prior stroke/TIA = 2 points
    score += int(vascular_disease)    # V: Vascular disease
    if sex == 'female':
        score += 1                    # Sc: Female sex = 1 point
    score = min(score, 9)             # Cap at 9
    risk  = STROKE_RISK.get(score, 15.2)
    return score, risk

# ============================================================
# ECG PLOT GENERATOR
# ============================================================

def generate_ecg_plot(signal, peaks, result_id, afib_prob):
    """
    Generates an ECG waveform plot with R-peaks marked and a
    color-coded background (red tint = high AFib probability).
    Saves as static/results/{result_id}_ecg.png.
    """
    fig, ax = plt.subplots(figsize=(14, 4))

    # Show up to 10 seconds
    n_show = min(int(10 * FS), len(signal))
    t      = np.arange(n_show) / FS
    sig    = signal[:n_show]

    # Background color based on risk
    if afib_prob >= 0.7:
        bg = '#fff0f0'   # Light red
    elif afib_prob >= 0.4:
        bg = '#fff8e8'   # Light amber
    else:
        bg = '#f0fff4'   # Light green

    ax.set_facecolor(bg)
    fig.patch.set_facecolor(bg)

    # Plot signal
    color = '#c0392b' if afib_prob >= 0.5 else '#1a5276'
    ax.plot(t, sig, color=color, linewidth=0.9)

    # Mark R-peaks
    vis_peaks = peaks[peaks < n_show]
    ax.scatter(vis_peaks / FS, sig[vis_peaks],
               color='orange', s=50, zorder=5, label='Detected heartbeat (R-peak)')

    ax.set_xlabel('Time (seconds)', fontsize=11)
    ax.set_ylabel('Amplitude (normalized)', fontsize=11)
    ax.set_title(f'ECG Recording  —  AFib Probability: {afib_prob*100:.1f}%',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(0, t[-1])

    out_path = f'static/results/{result_id}_ecg.png'
    plt.savefig(out_path, dpi=120, bbox_inches='tight', facecolor=bg)
    plt.close()
    return out_path


# ============================================================
# GRAD-CAM COMPUTATION
# ============================================================

def compute_gradcam(seg_input):
    # seg_input: shape (1, 9000, 1) float32 tensor ready for the model
    inp = tf.constant(seg_input, dtype=tf.float32)

    # Record operations so we can compute gradients afterward
    with tf.GradientTape() as tape:
        tape.watch(inp)
        conv_outputs, predictions = GRAD_MODEL(inp)  # forward pass
        afib_prob = predictions[:, 0]                # AFib probability

    # How much does each conv feature map affect the AFib probability?
    grads       = tape.gradient(afib_prob, conv_outputs)  # (1, timesteps, filters)
    pooled      = tf.reduce_mean(grads, axis=(0, 1))       # one score per filter

    # Weight each filter map by its importance and sum across filters
    conv_out    = conv_outputs[0]                           # (timesteps, filters)
    heatmap     = conv_out @ pooled[..., tf.newaxis]       # (timesteps, 1)
    heatmap     = tf.squeeze(heatmap).numpy()

    # ReLU: only keep regions that push toward AFib (positive contributions)
    heatmap     = np.maximum(heatmap, 0)
    heatmap     = heatmap / (heatmap.max() + 1e-8)         # normalize 0-1

    # Upsample from ~140 timesteps (conv output) back to 9000 (ECG length)
    heatmap_up  = np.interp(
        np.linspace(0, 1, SEG_LEN),
        np.linspace(0, 1, len(heatmap)),
        heatmap
    )
    return heatmap_up


def generate_gradcam_plot(signal, heatmap, result_id, afib_prob, predicted_label):
    # Generates ECG with red Grad-CAM attention heatmap overlaid.
    # Saves as static/results/{result_id}_gradcam.png
    fig, ax = plt.subplots(figsize=(14, 4))

    n_show    = min(int(10 * FS), len(signal))  # show first 10 seconds
    t         = np.arange(n_show) / FS
    sig       = signal[:n_show]
    heat_show = heatmap[:n_show]

    ax.set_facecolor('#fafafa')
    fig.patch.set_facecolor('#fafafa')

    # Plot ECG as dark line
    ax.plot(t, sig, color='#1a252f', linewidth=0.9, zorder=2, label='ECG signal')

    # Overlay red attention heatmap in chunks of 50 samples
    step = 50
    for i in range(0, n_show - step, step):
        alpha_val = float(heat_show[i:i+step].mean()) * 0.65
        if alpha_val > 0.03:
            ax.axvspan(t[i], t[i+step],
                       alpha=alpha_val, color='#e74c3c',
                       linewidth=0, zorder=1)

    ax.set_xlabel('Time (seconds)', fontsize=11)
    ax.set_ylabel('Amplitude (normalized)', fontsize=11)
    ax.set_title(
        f'CNN Attention Map (Grad-CAM)  |  Predicted: {predicted_label}  ({afib_prob*100:.1f}% AFib probability)  |  Red shading = regions the CNN focused on',
        fontsize=11, fontweight='bold'
    )
    ax.legend(fontsize=9, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(0, t[-1])

    out_path = f'static/results/{result_id}_gradcam.png'
    plt.savefig(out_path, dpi=120, bbox_inches='tight', facecolor='#fafafa')
    plt.close()
    return out_path

# ============================================================
# FLASK ROUTES
# ============================================================

@app.route('/')
def index():
    """Serve the main upload page."""
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Main analysis endpoint.
    Receives ECG file + patient info, runs models, returns results.
    """
    # ---- 1. Get patient info from the form ----
    try:
        age              = int(request.form.get('age', 65))
        sex              = request.form.get('sex', 'male').lower()
        hypertension     = 'hypertension'     in request.form
        diabetes         = 'diabetes'         in request.form
        prior_stroke     = 'prior_stroke'     in request.form
        heart_failure    = 'heart_failure'    in request.form
        vascular_disease = 'vascular_disease' in request.form
    except Exception as e:
        return jsonify({'error': f'Invalid patient data: {e}'}), 400

    # ---- 2. Save uploaded file ----
    if 'ecg_file' not in request.files:
        return jsonify({'error': 'No ECG file uploaded'}), 400

    f         = request.files['ecg_file']
    result_id = str(uuid.uuid4())[:8]   # Short unique ID for this analysis
    filename  = f'{result_id}_{f.filename}'
    filepath  = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    f.save(filepath)

    # ---- 3. Load and preprocess ECG ----
    try:
        raw_signal = load_ecg_from_file(filepath)
    except Exception as e:
        return jsonify({'error': f'Could not read ECG file: {e}'}), 400

    clean_signal = preprocess_ecg(raw_signal)
    peaks        = detect_r_peaks(clean_signal)

    # ---- 4. Run CNN-LSTM (primary model) ----
    if len(clean_signal) >= SEG_LEN:
        seg = clean_signal[:SEG_LEN]
    else:
        seg = np.concatenate([clean_signal, np.zeros(SEG_LEN - len(clean_signal))])

    seg_input  = seg.reshape(1, SEG_LEN, 1).astype(np.float32)
    cnn_prob   = float(CNN_MODEL.predict(seg_input, verbose=0)[0][0])
    gradcam_heatmap = compute_gradcam(seg_input)  # Grad-CAM heatmap

    # ---- 5. Run Random Forest (secondary model for cross-check) ----
    feats = extract_features(clean_signal)
    if feats is not None:
        feats_scaled = SCALER.transform([feats])
        rf_prob      = float(RF_MODEL.predict_proba(feats_scaled)[0][1])
    else:
        rf_prob = cnn_prob   # Fallback if signal too short

    # ---- 6. Final AFib probability = average of CNN and RF ----
    afib_prob = (cnn_prob + rf_prob) / 2.0

    # ---- 7. Risk category ----
    if afib_prob >= CNN_THRESH:
        afib_detected = True
        if afib_prob >= 0.80:
            risk_level = 'HIGH'
            risk_color = '#e74c3c'
            risk_bg    = '#fdf0f0'
        else:
            risk_level = 'MODERATE'
            risk_color = '#e67e22'
            risk_bg    = '#fdf6e8'
    else:
        afib_detected = False
        risk_level    = 'LOW'
        risk_color    = '#27ae60'
        risk_bg       = '#f0fdf4'

    # ---- 8. CHA₂DS₂-VASc score ----
    chads_score, annual_stroke_risk = calculate_chadsvasc(
        age, sex, hypertension, diabetes,
        prior_stroke, heart_failure, vascular_disease
    )

    # Composite risk: if AFib detected AND high CHA₂DS₂-VASc, escalate
    if afib_detected and chads_score >= 2:
        risk_level = 'HIGH'
        risk_color = '#e74c3c'
        risk_bg    = '#fdf0f0'

    # ---- 9. Heart rate estimate ----
    if len(peaks) >= 2:
        hr_bpm = len(peaks) / (len(clean_signal) / FS) * 60
    else:
        hr_bpm = None

    # ---- 10. Generate ECG plot ----
    ecg_plot_path = generate_ecg_plot(clean_signal, peaks, result_id, afib_prob)
    predicted_label = 'AFib Detected' if afib_prob >= CNN_THRESH else 'Normal Rhythm'
    gradcam_plot_path = generate_gradcam_plot(clean_signal, gradcam_heatmap, result_id, afib_prob, predicted_label)

    # ---- 11. Plain English explanation ----
    if risk_level == 'HIGH':
        explanation = (
            f"Your ECG shows strong signs of Atrial Fibrillation "
            f"({afib_prob*100:.0f}% probability). Combined with your clinical risk "
            f"factors (CHA\u2082DS\u2082-VASc score {chads_score}), your estimated "
            f"annual stroke risk is {annual_stroke_risk}%. "
            f"We strongly recommend consulting a cardiologist promptly."
        )
        recommendation = "Please consult a cardiologist as soon as possible."
    elif risk_level == 'MODERATE':
        explanation = (
            f"Your ECG shows possible signs of Atrial Fibrillation "
            f"({afib_prob*100:.0f}% probability). Your estimated annual stroke risk "
            f"based on clinical factors is {annual_stroke_risk}%. "
            f"Further evaluation is recommended."
        )
        recommendation = "Consider scheduling an appointment with a cardiologist."
    else:
        explanation = (
            f"Your ECG does not show significant signs of Atrial Fibrillation "
            f"({afib_prob*100:.0f}% probability). Your estimated annual stroke risk "
            f"is {annual_stroke_risk}%. Continue routine health monitoring."
        )
        recommendation = "No immediate action required. Maintain regular check-ups."

    # ---- 12. Return results ----
    return render_template('index.html',
        result        = True,
        risk_level    = risk_level,
        risk_color    = risk_color,
        risk_bg       = risk_bg,
        afib_prob_pct = f"{afib_prob*100:.1f}",
        afib_detected = afib_detected,
        cnn_prob_pct  = f"{cnn_prob*100:.1f}",
        rf_prob_pct   = f"{rf_prob*100:.1f}",
        chads_score   = chads_score,
        annual_stroke_risk = annual_stroke_risk,
        hr_bpm        = f"{hr_bpm:.0f}" if hr_bpm else "N/A",
        n_peaks       = len(peaks),
        ecg_plot      = ecg_plot_path,
        gradcam_plot  = gradcam_plot_path,
        explanation   = explanation,
        recommendation= recommendation,
        age=age, sex=sex,
    )


if __name__ == '__main__':
    print("Starting AFib Detection Web Portal...")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=True, port=5000)
