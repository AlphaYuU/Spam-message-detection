from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from spam_project.model_catalog import COMBINATION_COMPONENT_LABELS
from spam_project.ui.settings import (
    COLOR_HAM,
    COLOR_NEUTRAL,
    COLOR_RING_BACKGROUND,
    COLOR_SPAM,
    COLOR_TEXT_DEFAULT,
    COLOR_TEXT_MUTED,
    COMBINATION_CANVAS_SIZE,
    COMBINATION_RING_WIDTH,
    CONFIDENCE_CANVAS_SIZE,
    CONFIDENCE_RING_WIDTH,
    DETAIL_LABEL_WIDTH,
    STATUS_LABEL_WIDTH,
    STATUS_WRAP_LENGTH,
)


class ConfidenceRing(tk.Canvas):
    def __init__(
        self,
        master,
        size: int = CONFIDENCE_CANVAS_SIZE,
        ring_width: int = CONFIDENCE_RING_WIDTH,
        text_font=("Segoe UI", 14, "bold"),
        **kwargs,
    ):
        super().__init__(
            master,
            width=size,
            height=size,
            highlightthickness=0,
            **kwargs,
        )
        self.size = size
        self.ring_width = ring_width
        self.text_font = text_font
        self.set_score(0.0, None)

    def set_score(self, confidence, prediction_label=None):
        confidence = max(0.0, min(float(confidence or 0.0), 1.0))
        ring_color = self._get_ring_color(prediction_label)

        self.delete("all")
        padding = self.ring_width // 2 + 4
        bounds = (
            padding,
            padding,
            self.size - padding,
            self.size - padding,
        )
        self.create_oval(
            *bounds,
            outline=COLOR_RING_BACKGROUND,
            width=self.ring_width,
        )
        if confidence > 0:
            self.create_arc(
                *bounds,
                start=90,
                extent=-359.9 * confidence,
                style="arc",
                outline=ring_color,
                width=self.ring_width,
            )
        self.create_text(
            self.size // 2,
            self.size // 2,
            text=f"{confidence:.0%}",
            font=self.text_font,
            fill=ring_color,
        )

    @staticmethod
    def _get_ring_color(prediction_label):
        prediction_label = str(prediction_label or "").lower()
        if prediction_label == "spam":
            return COLOR_SPAM
        if prediction_label == "ham":
            return COLOR_HAM
        return COLOR_NEUTRAL


class SingleModelSummaryPanel(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.metric_value_labels = {}

        ttk.Label(self, text="Input Confidence").pack(anchor="center", padx=12, pady=(8, 2))
        self.ring = ConfidenceRing(self)
        self.ring.pack(padx=12, pady=(0, 8))

        metric_frame = ttk.Frame(self)
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

    def reset(self, metrics=None):
        self.ring.set_score(0.0, None)
        self.set_metrics(metrics or {})

    def set_prediction(self, result, metrics=None):
        self.ring.set_score(result.confidence, result.label)
        self.set_metrics(metrics or {})

    def set_metrics(self, metrics):
        for key, label in self.metric_value_labels.items():
            value = metrics.get(key)
            label.config(text="-" if value is None else f"{value:.2%}")


class CombinationSummaryPanel(ttk.Frame):
    def __init__(self, master, weights=None):
        super().__init__(master)
        self.rings = {}
        self.weight_labels = {}

        ttk.Label(self, text="Component Scores").pack(anchor="center", padx=12, pady=(8, 6))

        for component_key, component_label in COMBINATION_COMPONENT_LABELS.items():
            component_frame = ttk.Frame(self)
            component_frame.pack(anchor="center", padx=6, pady=(0, 10))

            ring = ConfidenceRing(
                component_frame,
                size=COMBINATION_CANVAS_SIZE,
                ring_width=COMBINATION_RING_WIDTH,
                text_font=("Segoe UI", 10, "bold"),
            )
            ring.pack(side="left", padx=(0, 12))

            text_frame = ttk.Frame(component_frame)
            text_frame.pack(side="left")
            ttk.Label(text_frame, text=component_label, font=("Segoe UI", 10), width=8).pack(anchor="w")
            weight_label = ttk.Label(
                text_frame,
                text="Weight: 0.00",
                font=("Segoe UI", 8),
                foreground=COLOR_TEXT_MUTED,
                width=12,
            )
            weight_label.pack(anchor="w")

            self.rings[component_key] = ring
            self.weight_labels[component_key] = weight_label

        self.set_weights(weights or {})
        self.reset()

    def reset(self):
        for ring in self.rings.values():
            ring.set_score(0.0, None)

    def set_prediction(self, components):
        for component_key, ring in self.rings.items():
            component = components.get(component_key)
            if component is None:
                ring.set_score(0.0, None)
                continue
            ring.set_score(component.confidence, component.label)

    def set_weights(self, weights):
        for component_key, label in self.weight_labels.items():
            label.config(text=f"Weight: {float(weights.get(component_key, 0.0)):.2f}")


class ResultPanel(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)

        ttk.Label(self, text="Detection Result:").pack(anchor="w", pady=6)
        result_frame = ttk.Frame(self, relief="sunken", borderwidth=1)
        result_frame.pack(fill="x", pady=(0, 12))

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
            foreground=COLOR_TEXT_MUTED,
            anchor="center",
            width=DETAIL_LABEL_WIDTH,
            wraplength=STATUS_WRAP_LENGTH,
        )
        self.detail_label.pack(pady=(0, 8))

    def set_loading(self):
        self.result_label.config(text="Detecting...", foreground=COLOR_TEXT_MUTED)
        self.detail_label.config(text="")

    def set_result(self, result, display_name):
        prediction_label = str(result.label).lower()
        if prediction_label == "spam":
            self.result_label.config(text="SPAM", foreground=COLOR_SPAM)
        else:
            self.result_label.config(text="HAM", foreground=COLOR_HAM)

        detail_text = f"Model: {display_name}"
        if result.confidence is not None:
            detail_text = f"{detail_text} | Confidence: {result.confidence:.2%}"
        self.detail_label.config(text=detail_text)

    def set_error(self, error_msg):
        self.result_label.config(text="Error", foreground=COLOR_TEXT_MUTED)
        self.detail_label.config(text=error_msg)

    def clear(self):
        self.result_label.config(text="-", foreground=COLOR_TEXT_DEFAULT)
        self.detail_label.config(text="")


class ExplanationTable(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)

        ttk.Label(self, text="Feature Explanation:").pack(anchor="w", pady=6)
        explanation_frame = ttk.Frame(self)
        explanation_frame.pack(fill="x", pady=(0, 12))

        explanation_columns = ("feature", "tfidf", "weight", "contribution", "tendency")
        self.table = ttk.Treeview(
            explanation_frame,
            columns=explanation_columns,
            show="headings",
            height=8,
        )
        self.table.heading("feature", text="Feature")
        self.table.heading("tfidf", text="TF-IDF")
        self.table.heading("weight", text="Weight")
        self.table.heading("contribution", text="Contribution")
        self.table.heading("tendency", text="Tendency")
        self.table.column("feature", width=210, anchor="w")
        self.table.column("tfidf", width=90, anchor="center")
        self.table.column("weight", width=90, anchor="center")
        self.table.column("contribution", width=110, anchor="center")
        self.table.column("tendency", width=90, anchor="center")
        self.table.tag_configure("spam", foreground=COLOR_SPAM)
        self.table.tag_configure("ham", foreground=COLOR_HAM)
        self.table.tag_configure("neutral", foreground=COLOR_TEXT_MUTED)
        self.table.pack(side="left", fill="x", expand=True)

        explanation_scrollbar = ttk.Scrollbar(
            explanation_frame,
            orient="vertical",
            command=self.table.yview,
        )
        self.table.configure(yscrollcommand=explanation_scrollbar.set)
        explanation_scrollbar.pack(side="right", fill="y")

        self.status_label = ttk.Label(
            self,
            text=" ",
            font=("Segoe UI", 9),
            foreground=COLOR_TEXT_MUTED,
            width=STATUS_LABEL_WIDTH,
            wraplength=STATUS_WRAP_LENGTH,
            justify="left",
        )
        self.status_label.pack(anchor="w", pady=(0, 12))

    def clear(self):
        for item in self.table.get_children():
            self.table.delete(item)

    def reset(self):
        self.clear()
        self.set_status("")

    def set_status(self, text):
        self.status_label.config(text=text or " ")

    def populate(self, explanations):
        self.clear()
        if not explanations:
            self.set_status("No known TF-IDF features found in this input.")
            return

        self.set_status(f"{len(explanations)} model features found")
        for item in explanations:
            tendency = str(item.tendency or "neutral").lower()
            self.table.insert(
                "",
                "end",
                values=(
                    item.feature,
                    self._format_number(item.tfidf_score),
                    self._format_number(item.model_weight),
                    self._format_number(item.contribution),
                    tendency.upper(),
                ),
                tags=(tendency,),
            )

    @staticmethod
    def _format_number(value):
        if value is None:
            return "-"
        return f"{float(value):.4f}"
