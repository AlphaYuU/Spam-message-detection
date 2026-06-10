import threading
import tkinter as tk
import math
from pathlib import Path
from tkinter import messagebox, scrolledtext, ttk

import joblib

from spam_feature_explainer import explain_linear_tfidf_model
from transformer_spam_model import predict_transformer_spam


PROJECT_ROOT = Path(__file__).resolve().parent
TRANSFORMER_MODEL_NAME = "XLM-RoBERTa"
STATUS_LABEL_WIDTH = 86
STATUS_WRAP_LENGTH = 680
DETAIL_LABEL_WIDTH = 64
CONFIDENCE_CANVAS_SIZE = 120
CONFIDENCE_RING_WIDTH = 18
MODEL_OPTIONS = {
    TRANSFORMER_MODEL_NAME: PROJECT_ROOT / "Models" / "transformer_spam_classifier",
    "Support Vector Machine": PROJECT_ROOT / "Models" / "svm_spam_classifier.joblib",
    "Logistic Regression": PROJECT_ROOT / "Models" / "logistic_regression_spam_classifier.joblib",
}
MODEL_METRICS = {
    TRANSFORMER_MODEL_NAME: {
        "accuracy": 0.988361,
        "precision": 0.975410,
        "recall": 0.929688,
    },
    "Support Vector Machine": {
        "accuracy": 0.984481,
        "precision": 0.982759,
        "recall": 0.890625,
    },
    "Logistic Regression": {
        "accuracy": 0.965082,
        "precision": 0.979167,
        "recall": 0.734375,
    },
}


class SpamDetectorApp:
    def __init__(self, root, model_options=None):
        self.root = root
        self.model_options = model_options or MODEL_OPTIONS

        self.root.title("Spam Detector")
        self.root.resizable(False, False)

        self.model_choice = tk.StringVar(value=next(iter(self.model_options)))

        self.text_input = None
        self.detect_btn = None
        self.result_label = None
        self.detail_label = None
        self.explanation_table = None
        self.explanation_status_label = None
        self.confidence_canvas = None
        self.confidence_text_id = None
        self.metric_value_labels = {}

        self._build_layout()
        self._lock_initial_window_size()

    def _build_layout(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill="x", padx=12, pady=(6, 0))

        input_frame = ttk.Frame(main_frame)
        input_frame.pack(side="left", fill="both", expand=True)

        selector_frame = ttk.Frame(input_frame)
        selector_frame.pack(fill="x", padx=12, pady=6)

        ttk.Label(selector_frame, text="Detection Model:").pack(side="left")
        model_dropdown = ttk.Combobox(
            selector_frame,
            textvariable=self.model_choice,
            values=list(self.model_options.keys()),
            state="readonly",
            width=28,
        )
        model_dropdown.pack(side="left", padx=(8, 0))
        model_dropdown.bind("<<ComboboxSelected>>", self.on_model_changed)

        ttk.Label(input_frame, text="Enter message text:").pack(anchor="w", padx=12, pady=6)
        self.text_input = scrolledtext.ScrolledText(
            input_frame,
            width=60,
            height=8,
            wrap=tk.WORD,
            font=("Segoe UI", 10),
        )
        self.text_input.pack(fill="x", padx=12)

        self.detect_btn = ttk.Button(input_frame, text="Detect", command=self.detect_spam)
        self.detect_btn.pack(pady=10)

        self._build_result_section(input_frame)
        self._build_model_summary_section(main_frame)
        self._build_explanation_section()

    def _build_result_section(self, parent):
        ttk.Label(parent, text="Detection Result:").pack(anchor="w", padx=12, pady=6)
        result_frame = ttk.Frame(parent, relief="sunken", borderwidth=1)
        result_frame.pack(fill="x", padx=12, pady=(0, 12))

        self.result_label = ttk.Label(
            result_frame,
            text="-",
            font=("Segoe UI", 13, "bold"),
            anchor="center",
        )
        self.result_label.pack(pady=8)

        self.detail_label = ttk.Label(
            result_frame,
            text="",
            font=("Segoe UI", 9),
            foreground="gray",
            anchor="center",
            width=DETAIL_LABEL_WIDTH,
            wraplength=STATUS_WRAP_LENGTH,
        )
        self.detail_label.pack(pady=(0, 8))

    def _build_model_summary_section(self, parent):
        summary_frame = ttk.Frame(parent)
        summary_frame.pack(side="right", fill="y", padx=(12, 12), pady=(6, 12))

        ttk.Label(
            summary_frame,
            text="Model Performance",
            font=("Segoe UI", 10),
        ).pack(anchor="w", padx=12, pady=(2, 8))

        ttk.Label(summary_frame, text="Input Confidence").pack(anchor="center", padx=12, pady=(8, 2))
        self.confidence_canvas = tk.Canvas(
            summary_frame,
            width=CONFIDENCE_CANVAS_SIZE,
            height=CONFIDENCE_CANVAS_SIZE,
            highlightthickness=0,
        )
        self.confidence_canvas.pack(padx=12, pady=(0, 8))

        metric_frame = ttk.Frame(summary_frame)
        metric_frame.pack(fill="x", padx=12, pady=(0, 8))

        for row, (key, label) in enumerate(
            (
                ("accuracy", "Accuracy"),
                ("precision", "Precision"),
                ("recall", "Recall"),
            )
        ):
            ttk.Label(metric_frame, text=f"{label}:").grid(row=row, column=0, sticky="w", pady=2)
            value_label = ttk.Label(metric_frame, text="-", font=("Segoe UI", 9, "bold"))
            value_label.grid(row=row, column=1, sticky="e", padx=(18, 0), pady=2)
            self.metric_value_labels[key] = value_label

        self.update_model_metrics()
        self.update_confidence_chart(0.0, None)

    def _build_explanation_section(self):
        ttk.Label(self.root, text="Feature Explanation:").pack(anchor="w", padx=12, pady=6)
        explanation_frame = ttk.Frame(self.root)
        explanation_frame.pack(fill="x", padx=12, pady=(0, 12))

        explanation_columns = ("feature", "tfidf", "weight", "contribution", "tendency")
        self.explanation_table = ttk.Treeview(
            explanation_frame,
            columns=explanation_columns,
            show="headings",
            height=8,
        )
        self.explanation_table.heading("feature", text="Feature")
        self.explanation_table.heading("tfidf", text="TF-IDF")
        self.explanation_table.heading("weight", text="Weight")
        self.explanation_table.heading("contribution", text="Contribution")
        self.explanation_table.heading("tendency", text="Tendency")
        self.explanation_table.column("feature", width=210, anchor="w")
        self.explanation_table.column("tfidf", width=90, anchor="center")
        self.explanation_table.column("weight", width=90, anchor="center")
        self.explanation_table.column("contribution", width=110, anchor="center")
        self.explanation_table.column("tendency", width=90, anchor="center")
        self.explanation_table.tag_configure("spam", foreground="#c0392b")
        self.explanation_table.tag_configure("ham", foreground="#27ae60")
        self.explanation_table.tag_configure("neutral", foreground="gray")
        self.explanation_table.pack(side="left", fill="x", expand=True)

        explanation_scrollbar = ttk.Scrollbar(
            explanation_frame,
            orient="vertical",
            command=self.explanation_table.yview,
        )
        self.explanation_table.configure(yscrollcommand=explanation_scrollbar.set)
        explanation_scrollbar.pack(side="right", fill="y")

        self.explanation_status_label = ttk.Label(
            self.root,
            text=" ",
            font=("Segoe UI", 9),
            foreground="gray",
            width=STATUS_LABEL_WIDTH,
            wraplength=STATUS_WRAP_LENGTH,
            justify="left",
        )
        self.explanation_status_label.pack(anchor="w", padx=12, pady=(0, 12))

    def _lock_initial_window_size(self):
        self.root.update_idletasks()
        width = self.root.winfo_reqwidth()
        height = self.root.winfo_reqheight()
        self.root.geometry(f"{width}x{height}")

    def detect_spam(self):
        message = self.text_input.get("1.0", tk.END).strip()
        if not message:
            messagebox.showwarning("Empty Input", "Please enter a message to detect.")
            return

        selected_model = self.model_choice.get()
        self.detect_btn.config(state="disabled")
        self.result_label.config(text="Detecting...", foreground="gray")
        self.detail_label.config(text="")
        self.set_explanation_status("")
        self.update_confidence_chart(0.0, None)
        self.clear_explanation_table()

        thread = threading.Thread(
            target=self._run_prediction,
            args=(message, selected_model),
            daemon=True,
        )
        thread.start()

    def _run_prediction(self, message, selected_model):
        try:
            model_path = self.model_options[selected_model]
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

            if selected_model == TRANSFORMER_MODEL_NAME:
                transformer_prediction = predict_transformer_spam(model_path, message)
                self.root.after(
                    0,
                    self.show_result,
                    transformer_prediction.label,
                    selected_model,
                    [],
                    transformer_prediction.confidence,
                    transformer_prediction.probabilities,
                )
                return

            model = joblib.load(model_path)
            prediction = model.predict([message])[0]
            explanations = explain_linear_tfidf_model(model, message)
            confidence, probabilities = self.get_classical_confidence(model, message)

            self.root.after(0, self.show_result, prediction, selected_model, explanations, confidence, probabilities)
        except Exception as error:
            self.root.after(0, self.show_error, str(error))

    def show_result(self, prediction, selected_model, explanations, confidence=None, probabilities=None):
        prediction_label = str(prediction).lower()
        if prediction_label == "spam":
            self.result_label.config(text="SPAM", foreground="#c0392b")
        else:
            self.result_label.config(text="HAM", foreground="#27ae60")

        detail_text = f"Model: {selected_model}"
        if confidence is not None:
            detail_text = f"{detail_text} | Confidence: {confidence:.2%}"
        self.detail_label.config(text=detail_text)
        self.update_model_metrics(selected_model)
        self.update_confidence_chart(confidence or 0.0, prediction_label)
        if explanations:
            self.populate_explanation_table(explanations)
        elif probabilities:
            self.clear_explanation_table()
            self.set_explanation_status(self.format_probability_status(probabilities))
        else:
            self.populate_explanation_table(explanations)
        self.detect_btn.config(state="normal")

    def show_error(self, error_msg):
        self.result_label.config(text="Error", foreground="gray")
        self.detail_label.config(text=error_msg)
        self.clear_explanation_table()
        self.set_explanation_status("")
        messagebox.showerror("Error", f"Detection failed:\n{error_msg}")
        self.detect_btn.config(state="normal")

    def on_model_changed(self, _event=None):
        self.update_model_metrics()
        self.update_confidence_chart(0.0, None)
        self.result_label.config(text="-", foreground="black")
        self.detail_label.config(text="")
        self.set_explanation_status("")
        self.clear_explanation_table()

    def update_model_metrics(self, selected_model=None):
        selected_model = selected_model or self.model_choice.get()
        metrics = MODEL_METRICS.get(selected_model, {})
        for key, label in self.metric_value_labels.items():
            value = metrics.get(key)
            label.config(text="-" if value is None else f"{value:.2%}")

    def update_confidence_chart(self, confidence, prediction_label):
        if self.confidence_canvas is None:
            return

        confidence = max(0.0, min(float(confidence or 0.0), 1.0))
        ring_color = "#7f8c8d"
        if prediction_label == "spam":
            ring_color = "#c0392b"
        elif prediction_label == "ham":
            ring_color = "#27ae60"

        self.confidence_canvas.delete("all")
        padding = CONFIDENCE_RING_WIDTH // 2 + 4
        bounds = (
            padding,
            padding,
            CONFIDENCE_CANVAS_SIZE - padding,
            CONFIDENCE_CANVAS_SIZE - padding,
        )
        self.confidence_canvas.create_oval(
            *bounds,
            outline="#e5e5e5",
            width=CONFIDENCE_RING_WIDTH,
        )
        if confidence > 0:
            self.confidence_canvas.create_arc(
                *bounds,
                start=90,
                extent=-359.9 * confidence,
                style="arc",
                outline=ring_color,
                width=CONFIDENCE_RING_WIDTH,
            )
        self.confidence_canvas.create_text(
            CONFIDENCE_CANVAS_SIZE // 2,
            CONFIDENCE_CANVAS_SIZE // 2,
            text=f"{confidence:.0%}",
            font=("Segoe UI", 14, "bold"),
            fill=ring_color,
        )

    def get_classical_confidence(self, model, message):
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba([message])[0]
            classes = [str(label).lower() for label in model.classes_]
            probability_by_label = {
                label: float(probability)
                for label, probability in zip(classes, probabilities)
            }
            return max(probability_by_label.values()), probability_by_label

        if hasattr(model, "decision_function"):
            score = float(model.decision_function([message])[0])
            confidence = 1.0 / (1.0 + math.exp(-abs(score)))
            classes = [str(label).lower() for label in model.classes_]
            if len(classes) == 2:
                positive_label = classes[1]
                negative_label = classes[0]
                if score >= 0:
                    probabilities = {
                        positive_label: confidence,
                        negative_label: 1.0 - confidence,
                    }
                else:
                    probabilities = {
                        negative_label: confidence,
                        positive_label: 1.0 - confidence,
                    }
                return confidence, probabilities

        return None, None

    def set_explanation_status(self, text):
        self.explanation_status_label.config(text=text or " ")

    def format_probability_status(self, probabilities):
        return " | ".join(
            f"{label.upper()}: {probability:.2%}"
            for label, probability in sorted(probabilities.items())
        )

    def clear_explanation_table(self):
        for item in self.explanation_table.get_children():
            self.explanation_table.delete(item)

    def populate_explanation_table(self, explanations):
        self.clear_explanation_table()
        if not explanations:
            self.set_explanation_status("No known TF-IDF features found in this input.")
            return

        self.set_explanation_status(f"{len(explanations)} model features found")
        for item in explanations:
            self.explanation_table.insert(
                "",
                "end",
                values=(
                    item.feature,
                    f"{item.tfidf_score:.4f}",
                    f"{item.model_weight:.4f}",
                    f"{item.contribution:.4f}",
                    item.tendency.upper(),
                ),
                tags=(item.tendency,),
            )


def create_app():
    root = tk.Tk()
    app = SpamDetectorApp(root)
    return root, app


def run_app():
    root, _ = create_app()
    root.mainloop()


if __name__ == "__main__":
    run_app()
