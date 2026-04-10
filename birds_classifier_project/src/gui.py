from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from .constants import CLASS_PAIRS, FEATURES
from .data_loader import BirdDataset
from .metrics import format_confusion_matrix
from .pipeline import ExperimentResult, run_experiment
from .visualization import create_decision_boundary_figure


class BirdsClassifierApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Birds Binary Classifier - Perceptron / Adaline")
        self.root.geometry("1280x820")
        self.root.minsize(1180, 760)

        self.current_result: ExperimentResult | None = None
        self.current_dataset: BirdDataset | None = None
        self.canvas_widget = None

        self.csv_path_var = tk.StringVar(value=str(Path("birds.csv")))
        self.feature_1_var = tk.StringVar(value=FEATURES[0])
        self.feature_2_var = tk.StringVar(value=FEATURES[1])
        self.class_pair_var = tk.StringVar(value="A & B")
        self.algorithm_var = tk.StringVar(value="perceptron")
        self.eta_var = tk.StringVar(value="0.01")
        self.epochs_var = tk.StringVar(value="100")
        self.mse_threshold_var = tk.StringVar(value="0.05")
        self.bias_var = tk.BooleanVar(value=True)
        self.seed_var = tk.StringVar(value="42")

        self.sample_feature_1_var = tk.StringVar()
        self.sample_feature_2_var = tk.StringVar()
        self.sample_result_var = tk.StringVar(value="Train a model first.")

        self._build_layout()
        self._refresh_sample_labels()

    def _build_layout(self) -> None:
        container = ttk.Frame(self.root, padding=10)
        container.pack(fill=tk.BOTH, expand=True)

        left_frame = ttk.Frame(container, padding=(0, 0, 12, 0))
        left_frame.pack(side=tk.LEFT, fill=tk.Y)

        right_frame = ttk.Frame(container)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self._build_controls(left_frame)
        self._build_results(right_frame)

    def _build_controls(self, parent: ttk.Frame) -> None:
        dataset_group = ttk.LabelFrame(parent, text="Dataset", padding=10)
        dataset_group.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(dataset_group, text="CSV Path").grid(row=0, column=0, sticky="w")
        ttk.Entry(dataset_group, textvariable=self.csv_path_var, width=38).grid(
            row=1, column=0, sticky="ew", padx=(0, 6)
        )
        ttk.Button(dataset_group, text="Browse", command=self._browse_csv).grid(
            row=1, column=1, sticky="ew"
        )
        dataset_group.columnconfigure(0, weight=1)

        selection_group = ttk.LabelFrame(parent, text="Selections", padding=10)
        selection_group.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(selection_group, text="Feature 1").grid(row=0, column=0, sticky="w")
        feature_1_combo = ttk.Combobox(
            selection_group,
            textvariable=self.feature_1_var,
            values=FEATURES,
            state="readonly",
            width=20,
        )
        feature_1_combo.grid(row=1, column=0, sticky="ew", padx=(0, 6))

        ttk.Label(selection_group, text="Feature 2").grid(row=0, column=1, sticky="w")
        feature_2_combo = ttk.Combobox(
            selection_group,
            textvariable=self.feature_2_var,
            values=FEATURES,
            state="readonly",
            width=20,
        )
        feature_2_combo.grid(row=1, column=1, sticky="ew")

        ttk.Label(selection_group, text="Class Pair").grid(
            row=2, column=0, sticky="w", pady=(8, 0)
        )
        class_combo = ttk.Combobox(
            selection_group,
            textvariable=self.class_pair_var,
            values=[f"{a} & {b}" for a, b in CLASS_PAIRS],
            state="readonly",
            width=20,
        )
        class_combo.grid(row=3, column=0, sticky="ew", padx=(0, 6))

        feature_1_combo.bind("<<ComboboxSelected>>", lambda _e: self._refresh_sample_labels())
        feature_2_combo.bind("<<ComboboxSelected>>", lambda _e: self._refresh_sample_labels())

        algorithm_group = ttk.LabelFrame(parent, text="Algorithm", padding=10)
        algorithm_group.pack(fill=tk.X, pady=(0, 8))

        ttk.Radiobutton(
            algorithm_group,
            text="Perceptron",
            variable=self.algorithm_var,
            value="perceptron",
        ).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(
            algorithm_group,
            text="Adaline",
            variable=self.algorithm_var,
            value="adaline",
        ).grid(row=0, column=1, sticky="w")

        params_group = ttk.LabelFrame(parent, text="Hyperparameters", padding=10)
        params_group.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(params_group, text="Learning Rate (eta)").grid(row=0, column=0, sticky="w")
        ttk.Entry(params_group, textvariable=self.eta_var, width=18).grid(
            row=1, column=0, sticky="ew", padx=(0, 6)
        )

        ttk.Label(params_group, text="Epochs (m)").grid(row=0, column=1, sticky="w")
        ttk.Entry(params_group, textvariable=self.epochs_var, width=18).grid(
            row=1, column=1, sticky="ew"
        )

        ttk.Label(params_group, text="MSE Threshold").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(params_group, textvariable=self.mse_threshold_var, width=18).grid(
            row=3, column=0, sticky="ew", padx=(0, 6)
        )

        ttk.Label(params_group, text="Random Seed").grid(row=2, column=1, sticky="w", pady=(8, 0))
        ttk.Entry(params_group, textvariable=self.seed_var, width=18).grid(
            row=3, column=1, sticky="ew"
        )

        ttk.Checkbutton(
            params_group,
            text="Use bias",
            variable=self.bias_var,
        ).grid(row=4, column=0, sticky="w", pady=(8, 0))

        actions_group = ttk.LabelFrame(parent, text="Actions", padding=10)
        actions_group.pack(fill=tk.X, pady=(0, 8))

        ttk.Button(actions_group, text="Train and Evaluate", command=self._train_and_evaluate).pack(
            fill=tk.X
        )

        classify_group = ttk.LabelFrame(parent, text="Single Sample Classification", padding=10)
        classify_group.pack(fill=tk.X, pady=(0, 8))

        self.sample_feature_1_label = ttk.Label(classify_group, text="Feature 1")
        self.sample_feature_1_label.grid(row=0, column=0, sticky="w")
        ttk.Entry(classify_group, textvariable=self.sample_feature_1_var).grid(
            row=1, column=0, sticky="ew", padx=(0, 6)
        )

        self.sample_feature_2_label = ttk.Label(classify_group, text="Feature 2")
        self.sample_feature_2_label.grid(row=0, column=1, sticky="w")
        ttk.Entry(classify_group, textvariable=self.sample_feature_2_var).grid(
            row=1, column=1, sticky="ew"
        )

        ttk.Button(classify_group, text="Classify Sample", command=self._classify_sample).grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0)
        )

        ttk.Label(
            classify_group,
            textvariable=self.sample_result_var,
            wraplength=350,
            justify=tk.LEFT,
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(8, 0))

        notes_group = ttk.LabelFrame(parent, text="Notes", padding=10)
        notes_group.pack(fill=tk.X)

        ttk.Label(
            notes_group,
            text=(
                "Gender input accepts: male / female / 1 / 0\n"
                "Rows are not dropped.\n"
                "Missing gender values are filled using the mode of the same class."
            ),
            wraplength=340,
            justify=tk.LEFT,
        ).pack(anchor="w")

    def _build_results(self, parent: ttk.Frame) -> None:
        top_group = ttk.LabelFrame(parent, text="Results", padding=8)
        top_group.pack(fill=tk.X, pady=(0, 8))

        self.results_text = tk.Text(top_group, height=16, wrap=tk.WORD)
        self.results_text.pack(fill=tk.X, expand=False)
        self.results_text.configure(state=tk.DISABLED)

        plot_group = ttk.LabelFrame(parent, text="Decision Boundary", padding=8)
        plot_group.pack(fill=tk.BOTH, expand=True)

        self.plot_container = ttk.Frame(plot_group)
        self.plot_container.pack(fill=tk.BOTH, expand=True)

    def _browse_csv(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Select birds.csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if file_path:
            self.csv_path_var.set(file_path)

    def _train_and_evaluate(self) -> None:
        try:
            selected_features = [self.feature_1_var.get(), self.feature_2_var.get()]
            if selected_features[0] == selected_features[1]:
                raise ValueError("Please select two different features.")

            class_pair = tuple(part.strip() for part in self.class_pair_var.get().split("&"))
            eta = float(self.eta_var.get())
            epochs = int(self.epochs_var.get())
            mse_threshold = float(self.mse_threshold_var.get())
            seed = int(self.seed_var.get())
            use_bias = self.bias_var.get()
            algorithm = self.algorithm_var.get()
            csv_path = self.csv_path_var.get().strip()

            self.current_dataset = BirdDataset(csv_path)
            self.current_result = run_experiment(
                csv_path=csv_path,
                selected_features=selected_features,
                class_pair=class_pair,
                algorithm=algorithm,
                eta=eta,
                epochs=epochs,
                mse_threshold=mse_threshold,
                use_bias=use_bias,
                seed=seed,
            )

            self._show_results(self.current_result)
            self._draw_plot(self.current_result)
            self.sample_result_var.set("Model trained. Enter a sample and click 'Classify Sample'.")
        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    def _show_results(self, result: ExperimentResult) -> None:
        confusion_text = format_confusion_matrix(result.confusion_matrix, result.split.class_pair)

        history_name = (
            "Misclassified samples per epoch"
            if self.algorithm_var.get() == "perceptron"
            else "MSE per epoch"
        )

        history_preview = result.model.history[:10]
        history_suffix = " ..." if len(result.model.history) > 10 else ""

        text = (
            f"Algorithm: {self.algorithm_var.get().title()}\n"
            f"Selected features: {result.split.feature_names[0]}, {result.split.feature_names[1]}\n"
            f"Selected classes: {result.split.class_pair[0]} vs {result.split.class_pair[1]}\n"
            f"Training samples: {len(result.split.X_train)} (30 per class)\n"
            f"Testing samples: {len(result.split.X_test)} (20 per class)\n"
            f"Epochs completed: {result.epochs_completed}\n"
            f"Weights: {result.model.weights}\n"
            f"Bias: {result.model.bias:.6f}\n"
            f"Decision equation:\n{result.decision_equation}\n\n"
            f"Confusion Matrix:\n{confusion_text}\n\n"
            f"Overall accuracy: {result.accuracy * 100:.2f}%\n\n"
            f"{history_name}:\n{history_preview}{history_suffix}\n"
        )

        self.results_text.configure(state=tk.NORMAL)
        self.results_text.delete("1.0", tk.END)
        self.results_text.insert(tk.END, text)
        self.results_text.configure(state=tk.DISABLED)

    def _draw_plot(self, result: ExperimentResult) -> None:
        for child in self.plot_container.winfo_children():
            child.destroy()

        figure = create_decision_boundary_figure(result)
        canvas = FigureCanvasTkAgg(figure, master=self.plot_container)
        canvas.draw()
        widget = canvas.get_tk_widget()
        widget.pack(fill=tk.BOTH, expand=True)
        self.canvas_widget = widget

    def _refresh_sample_labels(self) -> None:
        self.sample_feature_1_label.config(text=f"{self.feature_1_var.get()} value")
        self.sample_feature_2_label.config(text=f"{self.feature_2_var.get()} value")

    def _classify_sample(self) -> None:
        if self.current_result is None or self.current_dataset is None:
            messagebox.showinfo("Info", "Please train a model first.")
            return

        try:
            feature_names = self.current_result.split.feature_names
            value_1 = self.current_dataset.parse_single_feature_value(
                feature_names[0],
                self.sample_feature_1_var.get(),
            )
            value_2 = self.current_dataset.parse_single_feature_value(
                feature_names[1],
                self.sample_feature_2_var.get(),
            )

            prediction = self.current_result.model.predict([[value_1, value_2]])[0]
            predicted_class = self.current_result.split.label_mapping[int(prediction)]

            self.sample_result_var.set(
                f"Predicted class: {predicted_class} "
                f"(internal label {int(prediction)})"
            )
        except Exception as exc:
            messagebox.showerror("Classification Error", str(exc))


def launch_app() -> None:
    root = tk.Tk()
    app = BirdsClassifierApp(root)
    root.mainloop()
