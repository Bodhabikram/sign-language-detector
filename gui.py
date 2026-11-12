"""Desktop GUI for webcam-based ASL alphabet recognition and text composition."""

import json
import os
import queue
import threading
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import messagebox, ttk

from config import CAMERA_INDEX, CONFIDENCE_THRESHOLD, LABELS_PATH, MODEL_PATH
from preprocess_data import preprocess_single_image

WINDOW_TITLE = "ASL Sign Language Detector"
PREVIEW_SIZE = (800, 520)
SMOOTHING_WINDOW = 8
BOX_PADDING_RATIO = 0.25


class SignLanguageGUI:
    """Manage the desktop UI and a background webcam inference worker."""

    def __init__(self, root):
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.geometry("1120x760")
        self.root.minsize(900, 650)
        self.root.configure(bg="#0f172a")

        self._events = queue.Queue(maxsize=2)
        self._stop_event = threading.Event()
        self._worker = None
        self._photo = None
        self._current_label = None
        self._current_confidence = 0.0
        self._sentence = ""
        self._closing = False

        self.status_text = tk.StringVar(value="Ready. Start the camera to begin.")
        self.prediction_text = tk.StringVar(value="--")
        self.confidence_text = tk.StringVar(value="Show a hand sign to the camera")
        self.sentence_text = tk.StringVar(value="")

        self._configure_styles()
        self._build_layout()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(40, self._poll_worker_events)

    def _configure_styles(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(
            "TButton",
            font=("Segoe UI", 11, "bold"),
            padding=(14, 10),
            background="#334155",
            foreground="#f8fafc",
            borderwidth=0,
        )
        style.map("TButton", background=[("active", "#475569")])
        style.configure(
            "Accent.TButton",
            background="#2563eb",
            foreground="#ffffff",
        )
        style.map("Accent.TButton", background=[("active", "#1d4ed8")])
        style.configure(
            "Danger.TButton",
            background="#7f1d1d",
            foreground="#ffffff",
        )
        style.map("Danger.TButton", background=[("active", "#991b1b")])
        style.configure(
            "TEntry",
            fieldbackground="#1e293b",
            foreground="#f8fafc",
            padding=10,
        )

    def _build_layout(self):
        header = tk.Frame(self.root, bg="#0f172a")
        header.pack(fill="x", padx=28, pady=(24, 16))
        tk.Label(
            header,
            text="ASL Sign Detector",
            font=("Segoe UI", 25, "bold"),
            fg="#f8fafc",
            bg="#0f172a",
        ).pack(anchor="w")
        tk.Label(
            header,
            text="Show a letter, review the prediction, then add it to your text.",
            font=("Segoe UI", 11),
            fg="#94a3b8",
            bg="#0f172a",
        ).pack(anchor="w", pady=(5, 0))

        content = tk.Frame(self.root, bg="#0f172a")
        content.pack(fill="both", expand=True, padx=28, pady=(0, 16))
        content.columnconfigure(0, weight=7)
        content.columnconfigure(1, weight=4)
        content.rowconfigure(0, weight=1)

        preview_card = tk.Frame(content, bg="#111c31", highlightthickness=1,
                                highlightbackground="#263449")
        preview_card.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
        tk.Label(
            preview_card,
            text="LIVE CAMERA",
            font=("Segoe UI", 9, "bold"),
            fg="#94a3b8",
            bg="#111c31",
        ).pack(anchor="w", padx=16, pady=(14, 8))
        self.preview = tk.Label(
            preview_card,
            text="Camera is off\n\nPress Start camera",
            font=("Segoe UI", 16),
            fg="#94a3b8",
            bg="#020617",
            compound="center",
        )
        self.preview.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        side = tk.Frame(content, bg="#0f172a")
        side.grid(row=0, column=1, sticky="nsew")
        side.rowconfigure(2, weight=1)

        prediction_card = tk.Frame(side, bg="#111c31", highlightthickness=1,
                                   highlightbackground="#263449")
        prediction_card.pack(fill="x", pady=(0, 14))
        tk.Label(
            prediction_card,
            text="CURRENT SIGN",
            font=("Segoe UI", 9, "bold"),
            fg="#94a3b8",
            bg="#111c31",
        ).pack(anchor="w", padx=16, pady=(14, 6))
        tk.Label(
            prediction_card,
            textvariable=self.prediction_text,
            font=("Segoe UI", 46, "bold"),
            fg="#60a5fa",
            bg="#111c31",
        ).pack(anchor="w", padx=16)
        tk.Label(
            prediction_card,
            textvariable=self.confidence_text,
            font=("Segoe UI", 10),
            fg="#cbd5e1",
            bg="#111c31",
        ).pack(anchor="w", padx=16, pady=(0, 16))

        actions = tk.Frame(side, bg="#0f172a")
        actions.pack(fill="x", pady=(0, 14))
        self.start_button = ttk.Button(
            actions,
            text="Start camera",
            style="Accent.TButton",
            command=self.start_camera,
        )
        self.start_button.pack(side="left", fill="x", expand=True, padx=(0, 6))
        self.stop_button = ttk.Button(
            actions,
            text="Stop",
            style="Danger.TButton",
            command=self.stop_camera,
            state="disabled",
        )
        self.stop_button.pack(side="left", fill="x", expand=True, padx=(6, 0))

        text_card = tk.Frame(side, bg="#111c31", highlightthickness=1,
                             highlightbackground="#263449")
        text_card.pack(fill="both", expand=True)
        tk.Label(
            text_card,
            text="COMPOSED TEXT",
            font=("Segoe UI", 9, "bold"),
            fg="#94a3b8",
            bg="#111c31",
        ).pack(anchor="w", padx=16, pady=(14, 8))
        self.sentence_entry = ttk.Entry(
            text_card,
            textvariable=self.sentence_text,
            state="readonly",
            font=("Segoe UI", 16),
        )
        self.sentence_entry.pack(fill="x", padx=16, pady=(0, 12))

        compose_buttons = tk.Frame(text_card, bg="#111c31")
        compose_buttons.pack(fill="x", padx=16)
        ttk.Button(
            compose_buttons,
            text="Add current sign",
            style="Accent.TButton",
            command=self.add_current_sign,
        ).pack(fill="x", pady=4)
        edit_buttons = tk.Frame(text_card, bg="#111c31")
        edit_buttons.pack(fill="x", padx=12, pady=(4, 12))
        ttk.Button(edit_buttons, text="Space", command=self.add_space).pack(
            side="left", fill="x", expand=True, padx=4
        )
        ttk.Button(edit_buttons, text="Delete", command=self.delete_character).pack(
            side="left", fill="x", expand=True, padx=4
        )
        ttk.Button(edit_buttons, text="Clear", command=self.clear_text).pack(
            side="left", fill="x", expand=True, padx=4
        )
        tk.Label(
            text_card,
            text=(
                "Tip: pause on a sign, then click Add current sign. "
                "This avoids accidentally adding the same frame repeatedly."
            ),
            justify="left",
            wraplength=300,
            font=("Segoe UI", 9),
            fg="#94a3b8",
            bg="#111c31",
        ).pack(anchor="w", padx=16, pady=(0, 14))

        footer = tk.Frame(self.root, bg="#0b1220")
        footer.pack(fill="x", side="bottom")
        tk.Label(
            footer,
            textvariable=self.status_text,
            anchor="w",
            font=("Segoe UI", 10),
            fg="#cbd5e1",
            bg="#0b1220",
            padx=28,
            pady=12,
        ).pack(fill="x")

    @staticmethod
    def _get_hand_box(landmarks, frame_width, frame_height):
        x_values = [point.x * frame_width for point in landmarks.landmark]
        y_values = [point.y * frame_height for point in landmarks.landmark]
        x_min, x_max = min(x_values), max(x_values)
        y_min, y_max = min(y_values), max(y_values)
        box_width = x_max - x_min
        box_height = y_max - y_min
        x_min -= box_width * BOX_PADDING_RATIO
        x_max += box_width * BOX_PADDING_RATIO
        y_min -= box_height * BOX_PADDING_RATIO
        y_max += box_height * BOX_PADDING_RATIO
        side = max(x_max - x_min, y_max - y_min)
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        return (
            max(0, int(center_x - side / 2)),
            max(0, int(center_y - side / 2)),
            min(frame_width, int(center_x + side / 2)),
            min(frame_height, int(center_y + side / 2)),
        )

    def _publish(self, event):
        try:
            self._events.put_nowait(event)
        except queue.Full:
            try:
                self._events.get_nowait()
            except queue.Empty:
                pass
            try:
                self._events.put_nowait(event)
            except queue.Full:
                pass

    def start_camera(self):
        if self._worker is not None and self._worker.is_alive():
            return
        self._stop_event.clear()
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.status_text.set("Starting camera and loading the model...")
        self._worker = threading.Thread(
            target=self._camera_worker,
            name="asl-camera-worker",
            daemon=True,
        )
        self._worker.start()

    def stop_camera(self):
        self._stop_event.set()
        self.stop_button.configure(state="disabled")
        self.status_text.set("Stopping camera...")

    def _camera_worker(self):
        capture = None
        hand_tracker = None
        try:
            if not os.path.isfile(MODEL_PATH):
                raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
            if not os.path.isfile(LABELS_PATH):
                raise FileNotFoundError(f"Class labels not found: {LABELS_PATH}")

            with open(LABELS_PATH, "r", encoding="utf-8") as labels_file:
                label_mapping = json.load(labels_file)
            class_names = [
                label_mapping[str(index)] for index in range(len(label_mapping))
            ]
            model = tf.keras.models.load_model(MODEL_PATH)
            if model.output_shape[-1] != len(class_names):
                raise ValueError(
                    f"Model has {model.output_shape[-1]} outputs but the labels "
                    f"file has {len(class_names)} classes."
                )

            capture = cv2.VideoCapture(CAMERA_INDEX)
            if not capture.isOpened():
                raise RuntimeError(f"Could not open webcam index {CAMERA_INDEX}.")

            mp_hands = mp.solutions.hands
            mp_drawing = mp.solutions.drawing_utils
            hand_tracker = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.6,
            )
            recent_predictions = deque(maxlen=SMOOTHING_WINDOW)
            self._publish(("status", "Camera is running. Hold a sign in view."))

            while not self._stop_event.is_set():
                success, frame = capture.read()
                if not success or frame is None:
                    raise RuntimeError("The camera opened but failed to provide a frame.")

                model_frame = frame.copy()
                frame_height, frame_width = frame.shape[:2]
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hand_tracker.process(rgb_frame)
                prediction_label = None
                confidence = 0.0

                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                    )
                    x_min, y_min, x_max, y_max = self._get_hand_box(
                        hand_landmarks,
                        frame_width,
                        frame_height,
                    )

                    if x_max > x_min and y_max > y_min:
                        hand_crop = model_frame[y_min:y_max, x_min:x_max]
                        processed_bgr = preprocess_single_image(
                            hand_crop,
                            tracker=None,
                            isolate_hand=True,
                            enhance_levels=True,
                        )
                        processed_rgb = cv2.cvtColor(
                            processed_bgr,
                            cv2.COLOR_BGR2RGB,
                        )
                        model_input = np.expand_dims(
                            processed_rgb.astype(np.float32),
                            axis=0,
                        )
                        probabilities = model.predict(model_input, verbose=0)[0]
                        predicted_index = int(np.argmax(probabilities))
                        confidence = float(probabilities[predicted_index])
                        recent_predictions.append((predicted_index, confidence))

                        vote_counts = np.bincount(
                            [index for index, _ in recent_predictions],
                            minlength=len(class_names),
                        )
                        smoothed_index = int(np.argmax(vote_counts))
                        confidence = float(np.mean([
                            score
                            for index, score in recent_predictions
                            if index == smoothed_index
                        ]))
                        prediction_label = class_names[smoothed_index]
                        color = (
                            (40, 200, 100)
                            if confidence >= CONFIDENCE_THRESHOLD
                            else (0, 165, 255)
                        )
                        cv2.rectangle(
                            frame,
                            (x_min, y_min),
                            (x_max, y_max),
                            color,
                            2,
                        )
                    else:
                        recent_predictions.clear()
                else:
                    recent_predictions.clear()

                if prediction_label is None:
                    overlay = "No hand detected"
                    color = (0, 0, 255)
                else:
                    suffix = "" if confidence >= CONFIDENCE_THRESHOLD else " ?"
                    overlay = f"{prediction_label} ({confidence * 100:.1f}%){suffix}"
                cv2.putText(
                    frame,
                    overlay,
                    (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    color,
                    2,
                    cv2.LINE_AA,
                )
                self._publish(("frame", frame, prediction_label, confidence))
        except Exception as error:
            self._publish(("error", str(error)))
        finally:
            if capture is not None:
                capture.release()
            if hand_tracker is not None:
                hand_tracker.close()
            self._publish(("stopped", None))

    def _poll_worker_events(self):
        if self._closing:
            return
        try:
            while True:
                event = self._events.get_nowait()
                if event[0] == "status":
                    self.status_text.set(event[1])
                elif event[0] == "error":
                    self.status_text.set("Detection stopped because of an error.")
                    messagebox.showerror("Camera detection error", event[1], parent=self.root)
                elif event[0] == "stopped":
                    if not self._closing:
                        self.start_button.configure(state="normal")
                        self.stop_button.configure(state="disabled")
                        if self._stop_event.is_set():
                            self.status_text.set("Camera stopped.")
                elif event[0] == "frame":
                    _, frame, label, confidence = event
                    self._current_label = label
                    self._current_confidence = confidence
                    if label is None:
                        self.prediction_text.set("--")
                        self.confidence_text.set("No hand detected")
                    else:
                        self.prediction_text.set(label)
                        self.confidence_text.set(f"Confidence: {confidence * 100:.1f}%")
                    self._show_frame(frame)
        except queue.Empty:
            pass
        self.root.after(40, self._poll_worker_events)

    def _show_frame(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        image.thumbnail(PREVIEW_SIZE, Image.Resampling.LANCZOS)
        self._photo = ImageTk.PhotoImage(image=image)
        self.preview.configure(image=self._photo, text="")

    def _set_sentence(self):
        self.sentence_text.set(self._sentence)

    def add_current_sign(self):
        if self._current_label is None:
            self.status_text.set("No current prediction to add.")
            return
        if self._current_confidence < CONFIDENCE_THRESHOLD:
            self.status_text.set("Prediction confidence is low; hold the sign steady and try again.")
            return

        label = self._current_label
        if label == "nothing":
            self.status_text.set("No letter added for the 'nothing' class.")
        elif label == "space":
            self.add_space()
        elif label == "del":
            self.delete_character()
        elif len(label) == 1 and label.isalpha():
            self._sentence += label
            self._set_sentence()
            self.status_text.set(f"Added {label}.")
        else:
            self._sentence += label + " "
            self._set_sentence()
            self.status_text.set(f"Added {label}.")

    def add_space(self):
        if self._sentence and not self._sentence.endswith(" "):
            self._sentence += " "
            self._set_sentence()

    def delete_character(self):
        if self._sentence:
            self._sentence = self._sentence[:-1]
            self._set_sentence()

    def clear_text(self):
        self._sentence = ""
        self._set_sentence()
        self.status_text.set("Composed text cleared.")

    def _on_close(self):
        self._closing = True
        self._stop_event.set()
        if self._worker is not None and self._worker.is_alive():
            self.root.after(100, self._wait_for_worker)
        else:
            self.root.destroy()

    def _wait_for_worker(self):
        if self._worker is not None and self._worker.is_alive():
            self.root.after(100, self._wait_for_worker)
        else:
            self.root.destroy()

def main():
    root = tk.Tk()
    SignLanguageGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
