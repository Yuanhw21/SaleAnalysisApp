"""Data loading and preparation helpers for the retail sales dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.api import types as ptypes


SALES_FILENAME = "combined_data_202301.csv"
GEO_FILENAME = "qc_cities_coordinates.csv"

REQUIRED_SALES_COLUMNS: Tuple[str, ...] = (
    "Transaction_ID",
    "Contact_ID",
    "Date_transaction",
    "Store_transaction_ID",
    "Store_transaction_name",
    "Item_ID",
    "Unit_original_price",
    "Unit_sale_price",
    "Promotion_ID",
    "Quantity_item",
    "Transaction_value",
    "City",
    "Province",
    "Country",
    "LIBELLE_y",
    "STORE_ID",
    "COUNTRY",
    "PROVINCE",
    "CITY",
)

REQUIRED_GEO_COLUMNS: Tuple[str, ...] = ("City", "Latitude", "Longitude")


def _base_path(path_override: Optional[Path | str] = None) -> Path:
    if path_override is None:
        return Path(__file__).resolve().parent
    return Path(path_override).resolve()


def _coerce_numeric(series: pd.Series) -> pd.Series:
    """Convert numeric-looking strings that use commas as decimals to floats."""

    cleaned = (
        series.astype(str)
        .str.replace("\u00a0", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
        .replace({"nan": np.nan, "": np.nan})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _validate_columns(frame: pd.DataFrame, required: Iterable[str], frame_name: str) -> None:
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise KeyError(f"Missing required columns in {frame_name}: {missing}")


def load_raw_datasets(base_path: Optional[Path | str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    base = _base_path(base_path)
    sales_path = base / SALES_FILENAME
    geo_path = base / GEO_FILENAME

    if not sales_path.exists():
        raise FileNotFoundError(f"Sales data file not found: {sales_path}")
    if not geo_path.exists():
        raise FileNotFoundError(f"Geo data file not found: {geo_path}")

    sales = pd.read_csv(sales_path)
    geo = pd.read_csv(geo_path)

    _validate_columns(sales, REQUIRED_SALES_COLUMNS, "sales dataset")
    _validate_columns(geo, REQUIRED_GEO_COLUMNS, "geo dataset")

    return sales, geo


def prepare_geo_data(geo: pd.DataFrame) -> pd.DataFrame:
    cleaned = geo.copy()
    cleaned.columns = [col.strip() for col in cleaned.columns]
    cleaned["City"] = cleaned["City"].fillna("").astype(str).str.strip()
    cleaned["Latitude"] = _coerce_numeric(cleaned["Latitude"])
    cleaned["Longitude"] = _coerce_numeric(cleaned["Longitude"])
    return cleaned


def prepare_sales_data(sales: pd.DataFrame) -> pd.DataFrame:
    df = sales.copy()
    df.columns = [col.strip() for col in df.columns]

    df["Date_transaction"] = pd.to_datetime(df["Date_transaction"], format="%Y%m%d", errors="coerce")
    df = df[df["Date_transaction"].notna()]

    df["Year"] = df["Date_transaction"].dt.year
    df["Quarter"] = df["Date_transaction"].dt.quarter
    df["Month"] = df["Date_transaction"].dt.month

    df["Unit_original_price"] = _coerce_numeric(df["Unit_original_price"])
    df["Unit_sale_price"] = _coerce_numeric(df["Unit_sale_price"])
    df["Transaction_value"] = _coerce_numeric(df["Transaction_value"])
    df["Quantity_item"] = pd.to_numeric(df["Quantity_item"], errors="coerce")

    df["Item_ID"] = df["Item_ID"].astype(str).str.strip()
    df["Promotion_ID"] = df["Promotion_ID"].astype(str).str.strip()
    df["Promotion_ID"] = df["Promotion_ID"].replace({"nan": np.nan, "": np.nan})

    df["Line_Sales_Value"] = df["Unit_sale_price"] * df["Quantity_item"].fillna(0)
    df["Line_Original_Value"] = df["Unit_original_price"] * df["Quantity_item"].fillna(0)
    df["Line_Discount"] = df["Line_Original_Value"] - df["Line_Sales_Value"]

    df["CITY"] = df["CITY"].fillna("").astype(str).str.strip()
    df["days_since_epoch"] = (df["Date_transaction"] - pd.Timestamp("1970-01-01")).dt.days

    df["YearOfBirth"] = pd.to_numeric(df["YearOfBirth"], errors="coerce")
    df["Customer_Age"] = (
        df["Date_transaction"].dt.year - df["YearOfBirth"]
    )
    df.loc[df["Customer_Age"] < 0, "Customer_Age"] = np.nan

    df["Has_Web_Account"] = df["Has_Web_Account"].astype(str).str.strip().str.lower()
    df["Has_Web_Account"] = df["Has_Web_Account"].isin({"1", "true", "yes"})

    return df


def merge_sales_with_geo(sales: pd.DataFrame, geo: pd.DataFrame) -> pd.DataFrame:
    merged = sales.merge(geo, left_on="CITY", right_on="City", how="left", suffixes=("", "_geo"))

    online_mask = merged["LIBELLE_y"].isin([
        "0902 Ecom Website - LVER Canada",
        "0904 Ecom Website - LVER USA",
    ])
    merged.loc[online_mask, "CITY"] = "Online"

    if "Latitude_geo" in merged.columns:
        if "Latitude" in merged.columns:
            merged["Latitude"] = merged["Latitude"].fillna(merged["Latitude_geo"])
        else:
            merged.rename(columns={"Latitude_geo": "Latitude"}, inplace=True)

    if "Longitude_geo" in merged.columns:
        if "Longitude" in merged.columns:
            merged["Longitude"] = merged["Longitude"].fillna(merged["Longitude_geo"])
        else:
            merged.rename(columns={"Longitude_geo": "Longitude"}, inplace=True)

    drop_candidates = [col for col in ["City", "Latitude_geo", "Longitude_geo"] if col in merged]
    if drop_candidates:
        merged.drop(columns=drop_candidates, inplace=True)

    merged["Promo_Flag"] = merged["Promotion_ID"].notna()

    return merged


def load_and_prepare_data(base_path: Optional[Path | str] = None) -> pd.DataFrame:
    sales_raw, geo_raw = load_raw_datasets(base_path)
    geo = prepare_geo_data(geo_raw)
    sales = prepare_sales_data(sales_raw)
    merged = merge_sales_with_geo(sales, geo)
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

    transaction_level = data.drop_duplicates(subset="Transaction_ID")

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
    """Convenience wrapper for loading and computing helper tables."""

    base_path: Optional[Path | str] = None

    def run(self) -> pd.DataFrame:
        return load_and_prepare_data(self.base_path)

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
