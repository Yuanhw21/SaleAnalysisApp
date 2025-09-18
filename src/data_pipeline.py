"""Data loading and preparation helpers for the retail sales dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from .anonymize_datasets import (  # type: ignore
    ALBERTA_CITIES,
    build_city_mapping,
    hashed_postal,
    hashed_token,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
BASE_DIR = DATA_ROOT / "base"
DEFAULT_SALES_DIRS: Tuple[str, ...] = ("2023", "2024")


_MISSING_ID_TOKENS = {"", "na", "nan", "none", "null"}


def _normalise_identifier(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.strip()
    values = values.fillna("")
    mask = values.str.lower().isin(_MISSING_ID_TOKENS)
    return values.where(~mask, "")


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, dtype=str)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, dtype=str, engine="openpyxl")
    raise ValueError(f"Unsupported file extension for {path}")


def _coerce_numeric(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace("\u00a0", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
        .replace({"nan": np.nan, "": np.nan})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _list_sales_files(
    sales_dirs: Sequence[str] | None = None,
    months: Sequence[str] | None = None,
) -> Tuple[Path, ...]:
    dirs = sales_dirs or DEFAULT_SALES_DIRS
    files = []
    normalized_months = {m.lower() for m in months} if months else None
    for sub in dirs:
        year_path = DATA_ROOT / sub
        if not year_path.exists():
            continue
        year_candidates = {}
        for file_path in sorted(year_path.glob("Base_sales_*.*")):
            suffix = file_path.suffix.lower()
            if suffix not in {".csv", ".xlsx"}:
                continue
            month_key = file_path.stem.split("_")[-1].lower()
            if normalized_months and month_key not in normalized_months:
                continue
            current = year_candidates.get(month_key)
            if current is None or (current.suffix.lower() == ".xlsx" and suffix == ".csv"):
                year_candidates[month_key] = file_path
        for month_key in sorted(year_candidates):
            files.append(year_candidates[month_key])
    if not files:
        raise FileNotFoundError("No Base_sales_*.csv or .xlsx files found under data directory.")
    return tuple(files)


def list_available_months(sales_dirs: Sequence[str] | None = None) -> Tuple[str, ...]:
    files = _list_sales_files(sales_dirs, months=None)
    month_tokens = sorted({path.stem.split("_")[-1] for path in files})
    return tuple(month_tokens)


def _load_sales_frames(
    sales_dirs: Sequence[str] | None = None,
    months: Sequence[str] | None = None,
) -> pd.DataFrame:
    frames = []
    for file_path in _list_sales_files(sales_dirs, months):
        frame = _read_table(file_path)
        frame.columns = [col.strip() for col in frame.columns]
        frame["Source_File"] = file_path.name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def load_reference_tables() -> Dict[str, pd.DataFrame]:
    base_table_names = {
        "contact": "Base_contact",
        "item": "Base_item",
        "promotion": "Base_promotions",
        "store": "Base_stores",
    }

    tables: Dict[str, pd.DataFrame] = {}
    for key, base_name in base_table_names.items():
        candidates = [
            BASE_DIR / f"{base_name}.csv",
            BASE_DIR / f"{base_name}.xlsx",
        ]
        for path in candidates:
            if not path.exists():
                continue
            frame = _read_table(path)
            frame.columns = [col.strip() for col in frame.columns]
            tables[key] = frame
            break
        else:
            raise FileNotFoundError(
                f"Required base table not found: {candidates[0]} or {candidates[1]}"
            )
    return tables


def _merge_sales_with_references(
    sales: pd.DataFrame, tables: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    df = sales.copy()

    contact = tables["contact"].copy()
    item = tables["item"].copy()
    promotions = tables["promotion"].copy()
    stores = tables["store"].copy()

    # Normalise keys
    df["Contact_ID"] = _normalise_identifier(df["Contact_ID"])
    df["Item_ID"] = _normalise_identifier(df["Item_ID"])
    df["Promotion_ID"] = _normalise_identifier(df["Promotion_ID"])
    df["Store_transaction_ID"] = _normalise_identifier(df["Store_transaction_ID"])

    contact["Contact_ID"] = _normalise_identifier(contact["Contact_ID"])
    item["SKU"] = _normalise_identifier(item["SKU"])
    promotions["PROMO_ID"] = _normalise_identifier(promotions["PROMO_ID"])
    stores["STORE_ID"] = _normalise_identifier(stores["STORE_ID"])

    promotions.rename(columns={"LIBELLE": "LIBELLE_x"}, inplace=True)
    stores.rename(columns={"LIBELLE": "LIBELLE_y"}, inplace=True)

    df = df.merge(contact, on="Contact_ID", how="left")
    df = df.merge(item, how="left", left_on="Item_ID", right_on="SKU")
    df = df.merge(promotions, how="left", left_on="Promotion_ID", right_on="PROMO_ID")
    df = df.merge(stores, how="left", left_on="Store_transaction_ID", right_on="STORE_ID")

    return df


def _build_store_lookup(stores: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    store_df = stores.copy()
    store_df["STORE_ID"] = store_df["STORE_ID"].astype(str).str.strip()
    store_df["LIBELLE_y"] = store_df["LIBELLE_y"].fillna("")
    store_df["CITY"] = store_df["CITY"].fillna("")

    city_mapping = build_city_mapping(store_df["CITY"].tolist())
    store_lookup: Dict[str, Dict[str, Any]] = {}

    coords_map = {name: (lat, lon) for name, lat, lon in ALBERTA_CITIES}
    store_counter = 1
    online_counter = 1

    for _, row in store_df.drop_duplicates(subset="STORE_ID").iterrows():
        store_id = row["STORE_ID"]
        city_raw = row["CITY"]
        libelle = row["LIBELLE_y"]

        is_online = "ecom" in libelle.lower() or "online" in libelle.lower()
        if is_online:
            city_label = "Online"
            new_id = f"ABWEB{online_counter:02d}"
            new_name = f"Alberta Online Store {online_counter:02d}"
            lat = np.nan
            lon = np.nan
            online_counter += 1
        else:
            mapped = city_mapping.get(city_raw.strip().lower())
            if mapped:
                city_label, lat, lon = mapped
            else:
                city_label, lat, lon = ALBERTA_CITIES[0]
            new_id = f"AB{store_counter:03d}"
            new_name = f"AB Store {store_counter:03d} - {city_label}"
            lat = coords_map.get(city_label, (lat, lon))[0]
            lon = coords_map.get(city_label, (lat, lon))[1]
            store_counter += 1

        store_lookup[store_id] = {
            "id": new_id,
            "name": new_name,
            "city": city_label,
            "province": "AB",
            "country": "CAN",
            "postal": hashed_postal(new_id),
            "lat": lat,
            "lon": lon,
        }

    return store_lookup


def _apply_store_obfuscation(df: pd.DataFrame, store_lookup: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    store_id_series = df["STORE_ID"].fillna("").astype(str).str.strip()

    def lookup(key: str, field: str, fallback: Any = "") -> Any:
        info = store_lookup.get(key)
        return info.get(field) if info else fallback

    df["STORE_ID"] = store_id_series.apply(lambda x: lookup(x, "id", hashed_token(x, "ST") if x else ""))
    df["Store_transaction_ID"] = store_id_series.apply(lambda x: lookup(x, "id", hashed_token(x, "ST") if x else ""))
    df["Store_transaction_name"] = store_id_series.apply(
        lambda x: lookup(x, "name", hashed_token(x or "Store", "ABSTORE"))
    )
    df["LIBELLE_y"] = df["Store_transaction_name"]
    df["Store_Of_Creation"] = df["Store_Of_Creation"].fillna("").astype(str).str.strip().apply(
        lambda x: lookup(x, "id", hashed_token(x, "ST") if x else "")
    )
    df["CITY"] = store_id_series.apply(lambda x: lookup(x, "city", ""))
    df["PROVINCE"] = "AB"
    df["COUNTRY"] = "CAN"
    df["Latitude"] = store_id_series.apply(lambda x: lookup(x, "lat", np.nan))
    df["Longitude"] = store_id_series.apply(lambda x: lookup(x, "lon", np.nan))
    df["PostalCode"] = df["PostalCode"].apply(lambda v: hashed_postal(str(v)) if str(v).strip() else "")

    return df


def _apply_identifier_hashing(df: pd.DataFrame) -> pd.DataFrame:
    id_columns = {
        "Transaction_ID": "TRX",
        "Contact_ID": "CUST",
        "Item_ID": "ITEM",
        "Promotion_ID": "PROMO",
        "PROMO_ID": "PROMO",
        "SKU": "SKU",
        "EAN": "EAN",
        "LIBELLE_x": "OFFER",
    }

    for column, prefix in id_columns.items():
        if column not in df.columns:
            continue

        def obfuscate(value: Any) -> str:
            text = str(value).strip()
            return hashed_token(text, prefix) if text else ""

        df[column] = df[column].apply(obfuscate)

    return df


def _prepare_numeric_and_date_fields(df: pd.DataFrame) -> pd.DataFrame:
    df["Date_transaction"] = pd.to_datetime(df["Date_transaction"].astype(str), format="%Y%m%d", errors="coerce")
    df = df[df["Date_transaction"].notna()].copy()

    df["Unit_original_price"] = _coerce_numeric(df["Unit_original_price"])
    df["Unit_sale_price"] = _coerce_numeric(df["Unit_sale_price"])
    df["Transaction_value"] = _coerce_numeric(df["Transaction_value"])
    df["Quantity_item"] = pd.to_numeric(df["Quantity_item"], errors="coerce")

    df["Line_Sales_Value"] = df["Unit_sale_price"] * df["Quantity_item"].fillna(0)
    df["Line_Original_Value"] = df["Unit_original_price"] * df["Quantity_item"].fillna(0)
    df["Line_Discount"] = df["Line_Original_Value"] - df["Line_Sales_Value"]

    promo_ids = df["Promotion_ID"].astype("string").str.strip().fillna("")
    promo_missing = promo_ids.str.lower().isin(_MISSING_ID_TOKENS)
    df["Promo_Flag"] = (~promo_missing).astype(bool)

    df["Year"] = df["Date_transaction"].dt.year
    df["Quarter"] = df["Date_transaction"].dt.quarter
    df["Month"] = df["Date_transaction"].dt.month
    df["days_since_epoch"] = (df["Date_transaction"] - pd.Timestamp("1970-01-01")).dt.days

    df["YearOfBirth"] = pd.to_numeric(df["YearOfBirth"], errors="coerce")
    df["Customer_Age"] = df["Date_transaction"].dt.year - df["YearOfBirth"]
    df.loc[df["Customer_Age"] < 0, "Customer_Age"] = np.nan

    df["Has_Web_Account"] = df["Has_Web_Account"].astype(str).str.strip().str.lower()
    df["Has_Web_Account"] = df["Has_Web_Account"].isin({"1", "true", "yes", "y"})

    return df


def load_and_prepare_data(
    sales_dirs: Sequence[str] | None = None,
    months: Sequence[str] | None = None,
) -> pd.DataFrame:
    sales_raw = _load_sales_frames(sales_dirs, months)
    tables = load_reference_tables()
    merged = _merge_sales_with_references(sales_raw, tables)

    store_lookup = _build_store_lookup(tables["store"].rename(columns={"LIBELLE": "LIBELLE_y"}))
    merged = _apply_store_obfuscation(merged, store_lookup)
    merged = _apply_identifier_hashing(merged)
    merged = _prepare_numeric_and_date_fields(merged)
    if "Source_File" in merged.columns:
        merged.drop(columns=["Source_File"], inplace=True)

    return merged


def generate_data_quality_report(data: pd.DataFrame, focus_columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
    if focus_columns is not None:
        missing_cols = [col for col in focus_columns if col not in data.columns]
        if missing_cols:
            raise KeyError(f"Columns not found in dataset: {missing_cols}")
        subset = data.loc[:, focus_columns].copy()
    else:
        subset = data

    total_rows = len(subset)
    if total_rows == 0:
        return pd.DataFrame()

    zero_counts = {}
    for col in subset.columns:
        series = subset[col]
        if ptypes.is_numeric_dtype(series):
            zero_counts[col] = int((series == 0).sum())
        else:
            zero_counts[col] = np.nan

    summary = pd.DataFrame(
        {
            "dtype": subset.dtypes.astype(str),
            "missing_count": subset.isna().sum(),
            "missing_pct": (subset.isna().sum() / total_rows * 100).round(2),
            "zero_count": pd.Series(zero_counts),
            "unique_values": subset.nunique(dropna=True),
        }
    )

    return summary.sort_values("missing_pct", ascending=False)


def compute_highlights(data: pd.DataFrame) -> Dict[str, Any]:
    if data.empty:
        return {
            "total_revenue": 0.0,
            "total_units": 0.0,
            "total_orders": 0,
            "avg_order_value": 0.0,
            "top_city": None,
            "online_share": 0.0,
        }

    transaction_level = data.drop_duplicates(subset="Transaction_ID").copy()

    total_revenue = float(transaction_level["Transaction_value"].sum())
    total_orders = int(transaction_level["Transaction_ID"].nunique())
    total_units = float(data["Quantity_item"].sum())
    avg_order_value = float(total_revenue / total_orders) if total_orders else 0.0

    city_sales = (
        data.groupby("CITY")["Transaction_value"].sum().sort_values(ascending=False)
        if "CITY" in data.columns
        else pd.Series(dtype=float)
    )
    top_city = None
    if not city_sales.empty:
        top_city = (city_sales.index[0], float(city_sales.iloc[0]))

    online_share = 0.0
    if "CITY" in data.columns:
        online_revenue = data.loc[data["CITY"].str.lower() == "online", "Transaction_value"].sum()
        online_share = float(online_revenue / total_revenue) if total_revenue else 0.0

    return {
        "total_revenue": total_revenue,
        "total_units": total_units,
        "total_orders": total_orders,
        "avg_order_value": avg_order_value,
        "top_city": top_city,
        "online_share": online_share,
    }


def build_promotion_summary(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame()

    transaction_level = (
        data.groupby(["Transaction_ID", "Promo_Flag"])
        .agg(
            {
                "Transaction_value": "sum",
                "Line_Sales_Value": "sum",
                "Line_Discount": "sum",
                "Quantity_item": "sum",
            }
        )
        .reset_index()
    )

    grouped = transaction_level.groupby("Promo_Flag").agg(
        orders=("Transaction_ID", "nunique"),
        revenue=("Transaction_value", "sum"),
        avg_order_value=("Transaction_value", "mean"),
        total_discount=("Line_Discount", "sum"),
        avg_discount_per_order=("Line_Discount", "mean"),
        units=("Quantity_item", "sum"),
    )

    grouped.index = grouped.index.map({True: "With Promotion", False: "No Promotion"})
    grouped.index.name = "Promotion"
    return grouped


def build_profitability_by_store(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame()

    store_grouped = data.groupby(["LIBELLE_y", "CITY"], dropna=False).agg(
        revenue=("Transaction_value", "sum"),
        units=("Quantity_item", "sum"),
        discount=("Line_Discount", "sum"),
    )
    store_grouped["avg_discount_per_unit"] = store_grouped["discount"] / store_grouped["units"].replace(0, np.nan)
    store_grouped = store_grouped.sort_values("revenue", ascending=False)
    return store_grouped


def build_monthly_sales_series(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame(columns=["Month", "Revenue", "Orders", "Units"])

    transaction_level = data.drop_duplicates(subset="Transaction_ID")
    transaction_level["OrderMonth"] = transaction_level["Date_transaction"].dt.to_period("M").dt.to_timestamp()

    monthly = transaction_level.groupby("OrderMonth").agg(
        Revenue=("Transaction_value", "sum"),
        Orders=("Transaction_ID", "nunique"),
    )

    units = data.copy()
    units["OrderMonth"] = units["Date_transaction"].dt.to_period("M").dt.to_timestamp()
    monthly_units = units.groupby("OrderMonth")["Quantity_item"].sum()
    monthly["Units"] = monthly_units

    monthly = monthly.reset_index().rename(columns={"OrderMonth": "Month"})
    return monthly


def build_cohort_retention(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if data.empty:
        empty = pd.DataFrame()
        return empty, empty

    cohort_df = data.loc[data["Contact_ID"].notna(), ["Contact_ID", "Transaction_ID", "Date_transaction"]].copy()
    if cohort_df.empty:
        empty = pd.DataFrame()
        return empty, empty

    cohort_df["OrderMonth"] = cohort_df["Date_transaction"].dt.to_period("M").dt.to_timestamp()
    first_purchase = cohort_df.groupby("Contact_ID")["OrderMonth"].min().rename("CohortMonth")
    cohort_df = cohort_df.merge(first_purchase, on="Contact_ID", how="left")

    cohort_df["CohortIndex"] = (
        (cohort_df["OrderMonth"].dt.year - cohort_df["CohortMonth"].dt.year) * 12
        + (cohort_df["OrderMonth"].dt.month - cohort_df["CohortMonth"].dt.month)
        + 1
    )

    cohort_sizes = (
        cohort_df.groupby("CohortMonth")["Contact_ID"].nunique().rename("CohortSize")
    )

    retention = (
        cohort_df.groupby(["CohortMonth", "CohortIndex"])["Contact_ID"].nunique().unstack(fill_value=0)
    )

    retention = retention.divide(cohort_sizes, axis=0).round(3)

    return retention, cohort_sizes


@dataclass
class RetailDataPipeline:
    base_path: Optional[Path | str] = None
    sales_dirs: Sequence[str] | None = None
    months: Sequence[str] | None = None

    def run(self) -> pd.DataFrame:
        return load_and_prepare_data(self.sales_dirs, self.months)

    def summarize(self, data: pd.DataFrame) -> Dict[str, Any]:
        return compute_highlights(data)

    def promotion_summary(self, data: pd.DataFrame) -> pd.DataFrame:
        return build_promotion_summary(data)

    def profitability_by_store(self, data: pd.DataFrame) -> pd.DataFrame:
        return build_profitability_by_store(data)

    def monthly_sales(self, data: pd.DataFrame) -> pd.DataFrame:
        return build_monthly_sales_series(data)

    def cohorts(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return build_cohort_retention(data)
