# =========================
# model_example.py
# =========================
"""
Improved streaming training using SGDClassifier, stratified replay,
dynamic class weighting, PDE-inspired temporal regularization, and feature interactions.

Usage:
    python model_example_pa.py

Notes:
- It will try to import IncidentGenerator from data_generator.py and call `.generate()` or `.stream()`.
  If not available, it will fall back to reading streams_1k.json directly.
- Adjust constants below (BATCH_SIZE, REPLAY_MEMORY, REPLAY_RATIO) to taste.
"""

import json
import numpy as np
import random
from collections import deque, Counter
import time
import os

from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# --------------------
# Config
# --------------------
STREAM_PATH = "streams_1k.json"
BATCH_SIZE = 100
REPLAY_MEMORY = 1000   # larger memory for diversity
REPLAY_RATIO = 0.4     # fraction of combined batch drawn from replay
WARMUP = 100           # samples to warm up scaler
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# PDE-inspired hyperparam
PDE_ALPHA = 0.5        # weight of PDE temporal penalty when adjusting sample weights
PDE_SMOOTH = 0.1       # smoothing factor for predicted mean history

# --------------------
# Helper functions
# --------------------
def incident_to_features(inc):
    """Create base + interaction features from an incident dict."""
    severity = float(inc.get("severity", 1))
    cpu = float(inc.get("cpu_load", 0.0))
    net = float(inc.get("net_bytes", 0))
    # interactions / engineered
    interaction1 = cpu * severity
    interaction2 = cpu * np.log1p(net)
    return np.array([severity, cpu, net, interaction1, interaction2], dtype=float)

def incident_to_label(inc):
    critical = {"network_anomaly", "privilege_escalation", "policy_violation", "data_exfil", "malware_alert"}
    return 1 if inc.get("incident_type") in critical else 0

def stratified_sample_from_replay(replay_buffer, k):
    """Return X, y arrays sampled stratified by class from replay_buffer (list of (x,y))."""
    if k <= 0 or len(replay_buffer) == 0:
        return np.empty((0,)), np.empty((0,))
    by_class = {}
    for x, y in replay_buffer:
        by_class.setdefault(y, []).append((x, y))
    classes = sorted(by_class.keys())
    per_class = max(1, k // len(classes))
    sampled = []
    for cls in classes:
        pool = by_class[cls]
        if len(pool) <= per_class:
            sampled += pool
        else:
            sampled += random.sample(pool, per_class)
    # fill up if short
    while len(sampled) < k and len(replay_buffer) > 0:
        sampled.append(random.choice(replay_buffer))
    Xs = np.array([s[0] for s in sampled])
    ys = np.array([s[1] for s in sampled])
    return Xs, ys

def compute_balance_score(y_true, y_pred):
    """1 - |Recall_0 - Recall_1|, closer to 1 is more balanced."""
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    # rows=true classes
    with np.errstate(divide='ignore', invalid='ignore'):
        recall0 = cm[0,0] / cm[0].sum() if cm[0].sum() > 0 else 0.0
        recall1 = cm[1,1] / cm[1].sum() if cm[1].sum() > 0 else 0.0
    return 1.0 - abs(recall0 - recall1), recall0, recall1

# --------------------
# Try to import user's generator gracefully
# --------------------
use_generator = False
generator = None
try:
    from data_generator import IncidentGenerator
    try:
        # try to instantiate with common constructor signatures
        generator = IncidentGenerator(json_path=STREAM_PATH)
    except TypeError:
        try:
            generator = IncidentGenerator(data_path=STREAM_PATH)
        except TypeError:
            try:
                generator = IncidentGenerator()
            except Exception:
                generator = None
    if generator is not None:
        use_generator = True
except Exception:
    use_generator = False
    generator = None

# Fallback: read JSON file directly
if not use_generator:
    if not os.path.exists(STREAM_PATH):
        raise FileNotFoundError(f"Could not find generator and {STREAM_PATH} missing.")
    with open(STREAM_PATH, "r") as f:
        events_all = json.load(f)
    def stream_iter():
        for e in events_all:
            yield e
else:
    # try several method names
    if hasattr(generator, "generate"):
        def stream_iter():
            for e in generator.generate():
                yield e
    elif hasattr(generator, "stream"):
        def stream_iter():
            for e in generator.stream(n=len(getattr(generator, "events", [])), delay=0.0):
                yield e
    else:
        # as fallback, try iterating generator if it's an iterable
        try:
            def stream_iter():
                for e in generator:
                    yield e
        except Exception:
            # final fallback: load file
            with open(STREAM_PATH, "r") as f:
                events_all = json.load(f)
            def stream_iter():
                for e in events_all:
                    yield e

# --------------------
# Model, scaler, replay buffer
# --------------------
from sklearn.linear_model import SGDClassifier
model = SGDClassifier(loss="log_loss", learning_rate="optimal", random_state=42)

# model = PassiveAggressiveClassifier(C=0.01, max_iter=1000, random_state=SEED, tol=1e-3)
scaler = StandardScaler()
replay_buffer = deque(maxlen=REPLAY_MEMORY)

# warmup buffer for scaler
warmup = []

# bookkeeping for PDE temporal regularization
prev_mean_pred = None  # running previous mean prediction
smoothed_prev_mean = 0.5  # smoothed history

# training loop
X_batch_raw, y_batch = [], []
batch_count = 0
start_time = time.time()

for i, inc in enumerate(stream_iter()):
    x_raw = incident_to_features(inc)
    y = incident_to_label(inc)

    X_batch_raw.append(x_raw)
    y_batch.append(y)

    if len(warmup) < WARMUP:
        warmup.append(x_raw)

    # when batch ready or end
    if (i + 1) % BATCH_SIZE == 0:
        batch_count += 1
        Xb = np.vstack(X_batch_raw)
        yb = np.array(y_batch)

        # stratified replay sampling
        replay_k = int(REPLAY_RATIO * len(Xb))
        X_replay, y_replay = stratified_sample_from_replay(list(replay_buffer), replay_k)

        if X_replay.shape[0] > 0:
            X_comb = np.vstack([Xb, X_replay])
            y_comb = np.concatenate([yb, y_replay])
        else:
            X_comb = Xb.copy()
            y_comb = yb.copy()

        # feature scaling: fit scaler on warmup if not fitted
        if len(warmup) >= WARMUP and not hasattr(scaler, "mean_"):
            scaler.fit(np.vstack(warmup))
        if hasattr(scaler, "mean_"):
            X_comb_scaled = scaler.transform(X_comb)
            Xb_scaled = scaler.transform(Xb)
        else:
            X_comb_scaled = X_comb
            Xb_scaled = Xb

        # dynamic class weights -> sample weights
        classes_present = np.unique(y_comb)
        if len(classes_present) == 1:
            # if missing class, try to augment from replay (or jitter)
            missing_class = 0 if classes_present[0] == 1 else 1
            # try to get candidates from replay of missing class
            candidates = [r for r in list(replay_buffer) if r[1] == missing_class]
            if len(candidates) >= 1:
                # duplicate some
                add_n = min(len(candidates), 5)
                addX = np.array([c[0] for c in random.sample(candidates, add_n)])
                addy = np.array([c[1] for c in random.sample(candidates, add_n)])
                X_comb = np.vstack([X_comb, addX])
                y_comb = np.concatenate([y_comb, addy])
                # re-scale
                if hasattr(scaler, "mean_"):
                    X_comb_scaled = scaler.transform(X_comb)
            else:
                # jitter dominant class to synthesize minority examples (last resort)
                dom = classes_present[0]
                dom_idx = np.where(y_comb == dom)[0]
                synth_count = min(5, len(dom_idx))
                synth = []
                for _ in range(synth_count):
                    src = X_comb[ random.choice(dom_idx) ]
                    jitter = src + np.random.normal(scale=0.02, size=src.shape)
                    synth.append(jitter)
                if len(synth) > 0:
                    addX = np.vstack(synth)
                    addy = np.array([1-dom]*len(synth))
                    X_comb = np.vstack([X_comb, addX])
                    y_comb = np.concatenate([y_comb, addy])
                    if hasattr(scaler, "mean_"):
                        X_comb_scaled = scaler.transform(X_comb)

        # recompute classes present after augmentation
        classes_present = np.unique(y_comb)
        # compute balanced class weights (per-batch)
        try:
            cw = compute_class_weight(class_weight="balanced", classes=classes_present, y=y_comb)
            weight_dict = {c: w for c, w in zip(classes_present, cw)}
            sample_weights = np.array([weight_dict[yy] for yy in y_comb])
        except Exception:
            # fallback uniform
            sample_weights = np.ones(len(y_comb))

        # PDE-inspired temporal penalty: compute mean prediction change and adjust sample weights
        # get current mean prediction (on combined scaled data)
        try:
            preds_proba = None
            # PassiveAggressive doesn't provide predict_proba; use predict (0/1) mean as proxy
            cur_mean_pred = None
            if hasattr(model, "predict"):
                cur_mean_pred = np.mean(model.predict(X_comb_scaled)) if hasattr(model, "coef_") else 0.5
            else:
                cur_mean_pred = 0.5
        except Exception:
            cur_mean_pred = 0.5

        if prev_mean_pred is None:
            prev_mean_pred = cur_mean_pred
            smoothed_prev_mean = cur_mean_pred

        # compute temporal residual
        residual = cur_mean_pred - smoothed_prev_mean
        # update smoothed_prev_mean with PDE_SMOOTH
        smoothed_prev_mean = (1 - PDE_SMOOTH) * smoothed_prev_mean + PDE_SMOOTH * cur_mean_pred

        # apply penalty: if residual large, slightly reduce weights of currently dominant class
        adjust = np.exp(-PDE_ALPHA * (residual**2))
        sample_weights = sample_weights * adjust

        # final partial_fit (PassiveAggressive supports partial_fit)
        try:
            model.partial_fit(X_comb_scaled, y_comb, classes=np.array([0,1]), sample_weight=sample_weights)
        except Exception as e:
            # in case of any issue, try without sample_weight
            model.partial_fit(X_comb_scaled, y_comb, classes=np.array([0,1]))

        # update replay buffer with raw (unscaled) Xb and yb
        for xr, yr in zip(Xb, yb):
            replay_buffer.append((xr, yr))

        # evaluation on combined (quick)
        y_pred_comb = model.predict(X_comb_scaled)
        acc = accuracy_score(y_comb, y_pred_comb)
        bal_score, r0, r1 = compute_balance_score(y_comb, y_pred_comb)
        rep = classification_report(y_comb, y_pred_comb, digits=3, zero_division=0)
        cm = confusion_matrix(y_comb, y_pred_comb, labels=[0,1])

        print(f"\n--- Batch {batch_count} trained ---")
        print(f"Batch samples: {len(Xb)} | Combined: {len(X_comb)} | Replay size: {len(replay_buffer)}")
        print(f"Accuracy (combined): {acc:.3f} | Balance score: {bal_score:.3f} (recalls: {r0:.3f}, {r1:.3f})")
        print("Confusion matrix (rows=true, cols=pred):")
        print(cm)
        print("Classification report:")
        print(rep)
        print("-"*50)

        # reset
        X_batch_raw, y_batch = [], []

        # update prev_mean_pred for next iteration
        prev_mean_pred = cur_mean_pred

    # optional early stop for debugging
    if i > 5000:
        break

end_time = time.time()
print(f"Training finished in {end_time - start_time:.2f}s")

# Save final model & scaler
try:
    import joblib
    joblib.dump(model, "model_pa.joblib")
    joblib.dump(scaler, "scaler_pa.joblib")
    print("Saved model_pa.joblib and scaler_pa.joblib")
except Exception:
    pass
