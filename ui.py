import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, scrolledtext, ttk

import joblib

from spam_feature_explainer import explain_linear_tfidf_model


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_OPTIONS = {
    "Support Vector Machine": PROJECT_ROOT / "Models" / "svm_spam_classifier.joblib",
    "Logistic Regression": PROJECT_ROOT / "Models" / "logistic_regression_spam_classifier.joblib",
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

        self._build_layout()

    def _build_layout(self):
        selector_frame = ttk.Frame(self.root)
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

        ttk.Label(self.root, text="Enter message text:").pack(anchor="w", padx=12, pady=6)
        self.text_input = scrolledtext.ScrolledText(
            self.root,
            width=60,
            height=8,
            wrap=tk.WORD,
            font=("Segoe UI", 10),
        )
        self.text_input.pack(fill="x", padx=12)

        self.detect_btn = ttk.Button(self.root, text="Detect", command=self.detect_spam)
        self.detect_btn.pack(pady=10)

        self._build_result_section()
        self._build_explanation_section()

    def _build_result_section(self):
        ttk.Label(self.root, text="Detection Result:").pack(anchor="w", padx=12, pady=6)
        result_frame = ttk.Frame(self.root, relief="sunken", borderwidth=1)
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
        )
        self.detail_label.pack(pady=(0, 8))

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
            text="",
            font=("Segoe UI", 9),
            foreground="gray",
        )
        self.explanation_status_label.pack(anchor="w", padx=12, pady=(0, 12))

    def detect_spam(self):
        message = self.text_input.get("1.0", tk.END).strip()
        if not message:
            messagebox.showwarning("Empty Input", "Please enter a message to detect.")
            return

        selected_model = self.model_choice.get()
        self.detect_btn.config(state="disabled")
        self.result_label.config(text="Detecting...", foreground="gray")
        self.detail_label.config(text="")
        self.explanation_status_label.config(text="")
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

            model = joblib.load(model_path)
            prediction = model.predict([message])[0]
            explanations = explain_linear_tfidf_model(model, message)

            self.root.after(0, self.show_result, prediction, selected_model, explanations)
        except Exception as error:
            self.root.after(0, self.show_error, str(error))

    def show_result(self, prediction, selected_model, explanations):
        if str(prediction).lower() == "spam":
            self.result_label.config(text="SPAM", foreground="#c0392b")
        else:
            self.result_label.config(text="HAM", foreground="#27ae60")

        self.detail_label.config(text=f"Model: {selected_model}")
        self.populate_explanation_table(explanations)
        self.detect_btn.config(state="normal")

    def show_error(self, error_msg):
        self.result_label.config(text="Error", foreground="gray")
        self.detail_label.config(text=error_msg)
        self.clear_explanation_table()
        self.explanation_status_label.config(text="")
        messagebox.showerror("Error", f"Detection failed:\n{error_msg}")
        self.detect_btn.config(state="normal")

    def clear_explanation_table(self):
        for item in self.explanation_table.get_children():
            self.explanation_table.delete(item)

    def populate_explanation_table(self, explanations):
        self.clear_explanation_table()
        if not explanations:
            self.explanation_status_label.config(text="No known TF-IDF features found in this input.")
            return

        self.explanation_status_label.config(text=f"{len(explanations)} model features found")
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
