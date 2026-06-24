from __future__ import annotations

import threading
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk

from spam_project.model_catalog import (
    COMBINATION_COMPONENT_LABELS,
    COMBINATION_MODEL_NAME,
    DEFAULT_MODEL_DISPLAY_NAME,
)
from spam_project.models.registry import ModelRegistry
from spam_project.ui.formatters import format_combination_status, format_probability_status
from spam_project.ui.settings import SUMMARY_MIN_WIDTH
from spam_project.ui.widgets import (
    CombinationSummaryPanel,
    ExplanationTable,
    ResultPanel,
    SingleModelSummaryPanel,
)


class ModelController:
    def __init__(self, root, registry: ModelRegistry):
        self.root = root
        self.registry = registry
        self._active_request_id = 0
        self._lock = threading.Lock()

    def predict_async(self, model_key: str, message: str, on_success, on_error):
        request_id = self._next_request_id()
        thread = threading.Thread(
            target=self._run_prediction,
            args=(request_id, model_key, message, on_success, on_error),
            daemon=True,
        )
        thread.start()
        return thread

    def invalidate_pending(self):
        with self._lock:
            self._active_request_id += 1

    def _next_request_id(self):
        with self._lock:
            self._active_request_id += 1
            return self._active_request_id

    def _is_current_request(self, request_id: int) -> bool:
        with self._lock:
            return request_id == self._active_request_id

    def _run_prediction(self, request_id: int, model_key: str, message: str, on_success, on_error):
        try:
            result = self.registry.predict(model_key, message)
        except Exception as error:
            self._schedule_callback(request_id, on_error, str(error))
            return
        self._schedule_callback(request_id, on_success, result)

    def _schedule_callback(self, request_id: int, callback, *args):
        if not self._is_current_request(request_id) or not self._root_exists():
            return
        try:
            self.root.after(0, self._deliver_callback, request_id, callback, args)
        except tk.TclError:
            return

    def _deliver_callback(self, request_id: int, callback, args):
        if self._is_current_request(request_id) and self._root_exists():
            callback(*args)

    def _root_exists(self):
        try:
            return bool(self.root.winfo_exists())
        except tk.TclError:
            return False


class SpamDetectorApp:
    def __init__(self, root, registry: ModelRegistry | None = None):
        self.root = root
        self.model_registry = registry or ModelRegistry()
        self.controller = ModelController(root, self.model_registry)
        self.model_options = {
            display_name: self.model_registry.get_spec(display_name).path
            for display_name in self.model_registry.list_display_names()
        }

        self.root.title("Spam Detector")
        self.root.resizable(False, False)

        self.model_choice = tk.StringVar(value=DEFAULT_MODEL_DISPLAY_NAME)

        self.text_input = None
        self.detect_btn = None
        self.model_dropdown = None
        self.result_panel = None
        self.summary_title_label = None
        self.single_summary_panel = None
        self.combination_summary_panel = None
        self.explanation_widget = None

        self.result_label = None
        self.detail_label = None
        self.explanation_table = None
        self.explanation_status_label = None
        self.confidence_canvas = None
        self.combination_canvases = {}
        self.combination_weight_labels = {}
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
        self.model_dropdown = ttk.Combobox(
            selector_frame,
            textvariable=self.model_choice,
            values=self.model_registry.list_display_names(),
            state="readonly",
            width=28,
        )
        self.model_dropdown.pack(side="left", padx=(8, 0))
        self.model_dropdown.bind("<<ComboboxSelected>>", self.on_model_changed)

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

        self.result_panel = ResultPanel(input_frame)
        self.result_panel.pack(fill="x", padx=12)
        self.result_label = self.result_panel.result_label
        self.detail_label = self.result_panel.detail_label

        self._build_model_summary_section(main_frame)

        self.explanation_widget = ExplanationTable(self.root)
        self.explanation_widget.pack(fill="x", padx=12, pady=(0, 0))
        self.explanation_table = self.explanation_widget.table
        self.explanation_status_label = self.explanation_widget.status_label

    def _build_model_summary_section(self, parent):
        summary_frame = ttk.Frame(parent)
        summary_frame.pack(side="right", fill="y", padx=(12, 12), pady=(6, 12))
        ttk.Frame(summary_frame, width=SUMMARY_MIN_WIDTH, height=1).pack()

        self.summary_title_label = ttk.Label(
            summary_frame,
            text="Model Performance",
            font=("Segoe UI", 10),
        )
        self.summary_title_label.pack(anchor="w", padx=12, pady=(2, 8))

        self.single_summary_panel = SingleModelSummaryPanel(summary_frame)
        self.single_summary_panel.pack(fill="x")

        self.combination_summary_panel = CombinationSummaryPanel(
            summary_frame,
            weights=self._safe_get_combination_weights(),
        )

        self.confidence_canvas = self.single_summary_panel.ring
        self.metric_value_labels = self.single_summary_panel.metric_value_labels
        self.combination_canvases = self.combination_summary_panel.rings
        self.combination_weight_labels = self.combination_summary_panel.weight_labels
        self.update_summary_display()

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
        self.set_busy(True)
        self.result_panel.set_loading()
        self.explanation_widget.reset()
        self.reset_summary_for_prediction(selected_model)

        self.controller.predict_async(
            selected_model,
            message,
            self.show_result,
            self.show_error,
        )

    def show_result(self, result):
        display_name = result.metadata.get("display_name", self.model_choice.get())
        self.result_panel.set_result(result, display_name)

        if result.components:
            self.update_summary_display(COMBINATION_MODEL_NAME)
            self.combination_summary_panel.set_weights(result.metadata.get("weights", {}))
            self.combination_summary_panel.set_prediction(result.components)
            self.explanation_widget.clear()
            self.explanation_widget.set_status(format_combination_status(result))
        else:
            self.update_summary_display(display_name)
            self.single_summary_panel.set_prediction(
                result,
                metrics=self.model_registry.get_metrics(display_name),
            )
            if result.explanations:
                self.explanation_widget.populate(result.explanations)
            elif result.probabilities:
                self.explanation_widget.clear()
                self.explanation_widget.set_status(format_probability_status(result.probabilities))
            else:
                self.explanation_widget.populate(result.explanations)

        self.set_busy(False)

    def show_error(self, error_msg):
        self.result_panel.set_error(error_msg)
        self.explanation_widget.reset()
        messagebox.showerror("Error", f"Detection failed:\n{error_msg}")
        self.set_busy(False)

    def on_model_changed(self, _event=None):
        self.controller.invalidate_pending()
        self.update_summary_display()
        self.result_panel.clear()
        self.explanation_widget.reset()

    def reset_summary_for_prediction(self, selected_model=None):
        selected_model = selected_model or self.model_choice.get()
        self.update_summary_display(selected_model)
        spec = self.model_registry.get_spec(selected_model)
        if spec.kind == "combination":
            self.combination_summary_panel.reset()
        else:
            self.single_summary_panel.reset(metrics=self.model_registry.get_metrics(selected_model))

    def update_summary_display(self, selected_model=None):
        selected_model = selected_model or self.model_choice.get()
        spec = self.model_registry.get_spec(selected_model)
        is_combination_model = spec.kind == "combination"

        title = "Combination Model" if is_combination_model else "Model Performance"
        self.summary_title_label.config(text=title)

        if is_combination_model:
            if self.single_summary_panel.winfo_ismapped():
                self.single_summary_panel.pack_forget()
            if not self.combination_summary_panel.winfo_ismapped():
                self.combination_summary_panel.pack(fill="x")
            self.combination_summary_panel.set_weights(self._safe_get_combination_weights())
            self.combination_summary_panel.reset()
            return

        if self.combination_summary_panel.winfo_ismapped():
            self.combination_summary_panel.pack_forget()
        if not self.single_summary_panel.winfo_ismapped():
            self.single_summary_panel.pack(fill="x")
        self.single_summary_panel.set_metrics(self.model_registry.get_metrics(selected_model))

    def _safe_get_combination_weights(self):
        try:
            return self.model_registry.get_combination_weights(COMBINATION_MODEL_NAME)
        except (FileNotFoundError, KeyError, ValueError, OSError):
            return {
                component_key: 0.0
                for component_key in COMBINATION_COMPONENT_LABELS
            }

    def set_busy(self, is_busy: bool):
        button_state = "disabled" if is_busy else "normal"
        dropdown_state = "disabled" if is_busy else "readonly"
        if self.detect_btn is not None:
            self.detect_btn.config(state=button_state)
        if self.model_dropdown is not None:
            self.model_dropdown.config(state=dropdown_state)

    def set_explanation_status(self, text):
        self.explanation_widget.set_status(text)

    def clear_explanation_table(self):
        self.explanation_widget.clear()

    def populate_explanation_table(self, explanations):
        self.explanation_widget.populate(explanations)


def create_app():
    root = tk.Tk()
    app = SpamDetectorApp(root)
    return root, app


def run_app():
    root, _ = create_app()
    root.mainloop()
