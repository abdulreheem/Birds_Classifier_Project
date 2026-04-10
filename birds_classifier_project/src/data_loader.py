from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from .constants import CLASS_LABEL_COLUMN, FEATURES, GENDER_MAP


@dataclass
class DatasetSplit:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    feature_names: List[str]
    class_pair: Tuple[str, str]
    label_mapping: dict


class BirdDataset:
    def __init__(self, csv_path: str | Path) -> None:
        self.csv_path = Path(csv_path)
        self._df: pd.DataFrame | None = None

    @property
    def dataframe(self) -> pd.DataFrame:
        if self._df is None:
            self._df = self._load_and_preprocess()
        return self._df.copy()

    def _load_and_preprocess(self) -> pd.DataFrame:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        df = pd.read_csv(self.csv_path)

        required_columns = FEATURES + [CLASS_LABEL_COLUMN]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(
                f"CSV is missing required columns: {', '.join(missing_columns)}"
            )

        df = df.copy()
        df["gender"] = df["gender"].replace({"": np.nan})
        df["gender"] = df["gender"].apply(self._normalize_gender_text)

        # Fill missing genders using the mode of the same bird class.
        for class_name, class_df in df.groupby(CLASS_LABEL_COLUMN):
            class_gender_mode = class_df["gender"].dropna().mode()
            fill_value = class_gender_mode.iloc[0] if not class_gender_mode.empty else "female"
            class_mask = (df[CLASS_LABEL_COLUMN] == class_name) & (df["gender"].isna())
            df.loc[class_mask, "gender"] = fill_value

        # Encode gender numerically for use as a feature.
        df["gender"] = df["gender"].map(GENDER_MAP).astype(float)

        # Convert numeric features safely.
        numeric_features = [feature for feature in FEATURES if feature != "gender"]
        for feature in numeric_features:
            df[feature] = pd.to_numeric(df[feature], errors="raise")

        return df

    @staticmethod
    def _normalize_gender_text(value: object) -> str | float:
        if pd.isna(value):
            return np.nan
        text = str(value).strip().lower()
        if not text:
            return np.nan
        if text in {"m", "male", "1"}:
            return "male"
        if text in {"f", "female", "0"}:
            return "female"
        return np.nan

    def prepare_binary_split(
        self,
        selected_features: List[str],
        class_pair: Tuple[str, str],
        train_per_class: int = 30,
        seed: int = 42,
    ) -> DatasetSplit:
        if len(selected_features) != 2:
            raise ValueError("Exactly two features must be selected.")
        if selected_features[0] == selected_features[1]:
            raise ValueError("Feature 1 and Feature 2 must be different.")

        for feature in selected_features:
            if feature not in FEATURES:
                raise ValueError(f"Unsupported feature: {feature}")

        df = self.dataframe
        filtered = df[df[CLASS_LABEL_COLUMN].isin(class_pair)].copy()

        label_to_target = {
            class_pair[0]: -1,
            class_pair[1]: 1,
        }

        rng = np.random.default_rng(seed)
        train_frames = []
        test_frames = []

        for class_name in class_pair:
            class_df = filtered[filtered[CLASS_LABEL_COLUMN] == class_name].copy()
            if len(class_df) < train_per_class:
                raise ValueError(
                    f"Not enough samples in class {class_name} for the requested split."
                )

            shuffled_indices = rng.permutation(len(class_df))
            train_indices = shuffled_indices[:train_per_class]
            test_indices = shuffled_indices[train_per_class:]

            train_frames.append(class_df.iloc[train_indices].copy())
            test_frames.append(class_df.iloc[test_indices].copy())

        train_df = (
            pd.concat(train_frames, axis=0)
            .sample(frac=1, random_state=seed)
            .reset_index(drop=True)
        )
        test_df = (
            pd.concat(test_frames, axis=0)
            .sample(frac=1, random_state=seed + 1)
            .reset_index(drop=True)
        )

        X_train = train_df[selected_features].to_numpy(dtype=float)
        y_train = train_df[CLASS_LABEL_COLUMN].map(label_to_target).to_numpy(dtype=int)

        X_test = test_df[selected_features].to_numpy(dtype=float)
        y_test = test_df[CLASS_LABEL_COLUMN].map(label_to_target).to_numpy(dtype=int)

        return DatasetSplit(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            train_df=train_df,
            test_df=test_df,
            feature_names=selected_features,
            class_pair=class_pair,
            label_mapping={-1: class_pair[0], 1: class_pair[1]},
        )

    @staticmethod
    def parse_single_feature_value(feature_name: str, raw_value: str) -> float:
        if feature_name == "gender":
            normalized = BirdDataset._normalize_gender_text(raw_value)
            if pd.isna(normalized):
                raise ValueError(
                    "Gender must be one of: male, female, 1, or 0."
                )
            return float(GENDER_MAP[normalized])

        try:
            return float(raw_value)
        except ValueError as exc:
            raise ValueError(f"Invalid numeric value for {feature_name}: {raw_value}") from exc
