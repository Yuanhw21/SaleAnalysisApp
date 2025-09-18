from __future__ import annotations

import pandas as pd


def monthly_kpi_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Aggregate revenue, orders, units, and average order value by month."""
    if data.empty:
        return pd.DataFrame(columns=["Month", "Revenue", "Orders", "Units", "AvgOrderValue", "Unique_Customers"])

    frame = data.copy()
    frame["Month"] = frame["Date_transaction"].dt.to_period("M").dt.to_timestamp()

    summary = frame.groupby("Month").agg(
        Revenue=("Transaction_value", "sum"),
        Orders=("Transaction_ID", "nunique"),
        Units=("Quantity_item", "sum"),
        Unique_Customers=("Contact_ID", "nunique"),
    ).reset_index()
    summary["AvgOrderValue"] = summary["Revenue"] / summary["Orders"].replace(0, pd.NA)
    return summary


def category_revenue_share(data: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Return revenue share by merchandise category."""
    if "Catégorie" not in data.columns or data.empty:
        return pd.DataFrame(columns=["Catégorie", "Revenue", "Share"])

    totals = data.groupby("Catégorie")["Transaction_value"].sum().sort_values(ascending=False)
    total_revenue = totals.sum()
    result = totals.head(top_n).to_frame(name="Revenue")
    if total_revenue:
        result["Share"] = result["Revenue"] / total_revenue
    else:
        result["Share"] = 0
    return result.reset_index()


def promotion_activity_timeline(promotions: pd.DataFrame) -> pd.DataFrame:
    """Compute promotion durations and monthly activation counts."""
    if promotions.empty:
        return pd.DataFrame(columns=["Month", "ActivePromotions"])

    frame = promotions.copy()
    frame["START_DATE"] = pd.to_datetime(frame["START_DATE"], errors="coerce")
    frame["END_DATE"] = pd.to_datetime(frame["END_DATE"], errors="coerce")
    frame["END_DATE"].fillna(pd.Timestamp.today(), inplace=True)

    timeline = []
    for _, row in frame.iterrows():
        start = row["START_DATE"]
        end = row["END_DATE"]
        if pd.isna(start) or pd.isna(end):
            continue
        periods = pd.period_range(start=start, end=end, freq="M")
        for period in periods:
            timeline.append({"Month": period.to_timestamp(), "PROMO_ID": row["PROMO_ID"]})

    if not timeline:
        return pd.DataFrame(columns=["Month", "ActivePromotions"])

    timeline_df = pd.DataFrame(timeline)
    summary = timeline_df.groupby("Month")["PROMO_ID"].nunique().reset_index()
    summary.rename(columns={"PROMO_ID": "ActivePromotions"}, inplace=True)
    return summary


def store_opening_trend(stores: pd.DataFrame) -> pd.DataFrame:
    """Count store openings by year."""
    if stores.empty:
        return pd.DataFrame(columns=["Year", "Openings"])

    frame = stores.copy()
    frame["OPEN_DATE"] = pd.to_datetime(frame["OPEN_DATE"], errors="coerce")
    frame = frame[frame["OPEN_DATE"].notna()]
    frame["Year"] = frame["OPEN_DATE"].dt.year
    summary = frame.groupby("Year").size().reset_index(name="Openings")
    return summary.sort_values("Year")


def web_account_share(contacts: pd.DataFrame) -> pd.Series:
    """Calculate the share of customers with web accounts."""
    if contacts.empty or "Has_Web_Account" not in contacts.columns:
        return pd.Series(dtype=float)

    frame = contacts.copy()
    frame["Has_Web_Account"] = frame["Has_Web_Account"].astype(str).str.upper()
    counts = frame["Has_Web_Account"].value_counts(dropna=False)
    return counts / counts.sum()


import numpy as np
from scipy import stats


def prepare_transaction_level(data: pd.DataFrame) -> pd.DataFrame:
    """Aggregate line-level sales into transaction level for promo analysis."""
    if data.empty:
        return pd.DataFrame(columns=[
            "Transaction_ID",
            "Transaction_value",
            "Quantity_item",
            "Promo_Flag",
        ])

    grouped = (
        data.groupby("Transaction_ID").agg(
            Transaction_value=("Transaction_value", "sum"),
            Quantity_item=("Quantity_item", "sum"),
            Promo_Flag=("Promo_Flag", "max"),
        )
    ).reset_index()
    grouped["Promo_Flag"] = grouped["Promo_Flag"].astype(bool)
    return grouped


def run_promo_regression(transactions: pd.DataFrame) -> pd.DataFrame:
    """Run a simple OLS regression of order value on promotion flag."""
    if transactions.empty:
        return pd.DataFrame(columns=["term", "coef", "std_err", "t", "p_value"])

    df = transactions.dropna(subset=["Transaction_value", "Promo_Flag"]).copy()
    if df.empty:
        return pd.DataFrame(columns=["term", "coef", "std_err", "t", "p_value"])

    y = df["Transaction_value"].astype(float).values
    x = df["Promo_Flag"].astype(int).values
    X = np.column_stack([np.ones_like(x), x])

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    resid = y - y_hat
    n = len(y)
    k = X.shape[1]
    df_resid = n - k
    if df_resid <= 0:
        return pd.DataFrame(columns=["term", "coef", "std_err", "t", "p_value"])

    sigma2 = (resid @ resid) / df_resid
    cov = sigma2 * np.linalg.inv(X.T @ X)
    std_err = np.sqrt(np.diag(cov))
    t_vals = beta / std_err
    p_vals = 2 * (1 - stats.t.cdf(np.abs(t_vals), df_resid))

    terms = ["Intercept", "Promo_Flag"]
    return pd.DataFrame({
        "term": terms,
        "coef": beta,
        "std_err": std_err,
        "t": t_vals,
        "p_value": p_vals,
    })


def ab_test_promo(transactions: pd.DataFrame) -> dict:
    """Perform a simple A/B test comparing promo vs non-promo order values."""
    if transactions.empty:
        return {
            "promo_mean": np.nan,
            "nonpromo_mean": np.nan,
            "lift": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
        }

    df = transactions.dropna(subset=["Transaction_value", "Promo_Flag"]).copy()
    promo_values = df.loc[df["Promo_Flag"], "Transaction_value"].astype(float)
    nonpromo_values = df.loc[~df["Promo_Flag"], "Transaction_value"].astype(float)

    if promo_values.empty or nonpromo_values.empty:
        return {
            "promo_mean": promo_values.mean() if not promo_values.empty else np.nan,
            "nonpromo_mean": nonpromo_values.mean() if not nonpromo_values.empty else np.nan,
            "lift": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
        }

    promo_mean = promo_values.mean()
    nonpromo_mean = nonpromo_values.mean()
    lift = promo_mean - nonpromo_mean
    t_stat, p_value = stats.ttest_ind(promo_values, nonpromo_values, equal_var=False)

    return {
        "promo_mean": promo_mean,
        "nonpromo_mean": nonpromo_mean,
        "lift": lift,
        "t_stat": t_stat,
        "p_value": p_value,
    }
