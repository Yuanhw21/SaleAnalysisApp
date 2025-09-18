import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
from scipy.stats import chi2_contingency

from src.data_pipeline import (
    load_and_prepare_data,
    compute_highlights,
    generate_data_quality_report,
    build_promotion_summary,
    build_profitability_by_store,
    build_monthly_sales_series,
    build_cohort_retention,
    list_available_months,
)


st.set_page_config(page_title="Retail Sales Intelligence", layout="wide")


def quantile_rank_score(series: pd.Series, desired_bins: int = 5, reverse: bool = False) -> pd.Series:
    """Assign quantile-based scores (1-desired_bins) with safe fallbacks for small datasets."""
    if series.empty:
        return pd.Series(dtype=int)

    ranked = series.rank(method="first", ascending=not reverse)
    unique_values = ranked.nunique()
    num_bins = min(desired_bins, unique_values)

    if num_bins <= 1:
        neutral_score = desired_bins // 2 + 1
        return pd.Series(neutral_score, index=series.index, dtype=int)

    bins = pd.qcut(ranked, num_bins, labels=False)
    scores = bins + 1

    if num_bins < desired_bins:
        scores = np.ceil(scores * desired_bins / num_bins)

    if reverse:
        scores = desired_bins + 1 - scores

    return scores.astype(int)


def format_currency(value: float) -> str:
    if value is None or np.isnan(value):
        return "$0"
    return f"${value:,.0f}"


def format_percent(value: float) -> str:
    if value is None or np.isnan(value):
        return "0%"
    return f"{value:.0%}"


@st.cache_data
def load_data(months: tuple[str, ...]) -> pd.DataFrame:
    return load_and_prepare_data(months=months)


def apply_sidebar_filters(data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    filtered = data.copy()
    selections: dict = {}

    st.sidebar.header("Filters")

    years = sorted(filtered["Year"].dropna().unique())
    selected_years = st.sidebar.multiselect("Year", options=years, format_func=lambda x: str(int(x)))
    if selected_years:
        filtered = filtered[filtered["Year"].isin(selected_years)]
    selections["years"] = selected_years

    quarters = sorted(filtered["Quarter"].dropna().unique())
    selected_quarters = st.sidebar.multiselect("Quarter", options=quarters, format_func=lambda x: f"Q{int(x)}")
    if selected_quarters:
        filtered = filtered[filtered["Quarter"].isin(selected_quarters)]
    selections["quarters"] = selected_quarters

    months = sorted(filtered["Month"].dropna().unique())
    selected_months = st.sidebar.multiselect("Month", options=months, format_func=lambda x: f"{int(x):02d}")
    if selected_months:
        filtered = filtered[filtered["Month"].isin(selected_months)]
    selections["months"] = selected_months

    days_series = data["days_since_epoch"].dropna().astype(int)
    min_day = int(days_series.min()) if not days_series.empty else 0
    max_day = int(days_series.max()) if not days_series.empty else 0

    filtered_days = filtered["days_since_epoch"].dropna().astype(int)
    default_start = int(filtered_days.min()) if not filtered_days.empty else min_day
    default_end = int(filtered_days.max()) if not filtered_days.empty else max_day

    epoch = pd.Timestamp("1970-01-01")
    if min_day == max_day:
        selected_days = (min_day, max_day)
        st.sidebar.info("Only one day of data is available for the selected filters.")
    else:
        selected_days = st.sidebar.slider(
            "Date range",
            min_value=min_day,
            max_value=max_day,
            value=(default_start, default_end),
        )

    filtered = filtered[
        (filtered["days_since_epoch"] >= selected_days[0])
        & (filtered["days_since_epoch"] <= selected_days[1])
    ]
    start_date = (epoch + pd.to_timedelta(selected_days[0], unit="D")).strftime("%Y-%m-%d")
    end_date = (epoch + pd.to_timedelta(selected_days[1], unit="D")).strftime("%Y-%m-%d")
    st.sidebar.caption(f"Date range: {start_date} → {end_date}")
    selections["days"] = (start_date, end_date)

    categories = sorted(filtered["Catégorie"].dropna().unique())
    selected_categories = st.sidebar.multiselect("Category", options=categories)
    if selected_categories:
        filtered = filtered[filtered["Catégorie"].isin(selected_categories)]
    selections["categories"] = selected_categories

    item_ids = sorted(filtered["Item_ID"].dropna().unique())
    selected_items = st.sidebar.multiselect("Item ID", options=item_ids)
    if selected_items:
        filtered = filtered[filtered["Item_ID"].isin(selected_items)]
    selections["items"] = selected_items

    cities = sorted(filtered["CITY"].dropna().unique())
    selected_cities = st.sidebar.multiselect("City", options=cities)
    if selected_cities:
        filtered = filtered[filtered["CITY"].isin(selected_cities)]
    selections["cities"] = selected_cities

    if selected_cities:
        store_options = filtered[filtered["CITY"].isin(selected_cities)]["LIBELLE_y"].dropna().unique()
    else:
        store_options = filtered["LIBELLE_y"].dropna().unique()
    store_options = sorted(store_options)

    selected_stores = st.sidebar.multiselect("Store", options=store_options)
    if selected_stores:
        filtered = filtered[filtered["LIBELLE_y"].isin(selected_stores)]
    selections["stores"] = selected_stores

    return filtered, selections


def render_kpi_summary(highlights: dict) -> None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Revenue", format_currency(highlights.get("total_revenue", 0.0)))
    col2.metric("Orders", f"{highlights.get('total_orders', 0):,}")
    col3.metric("Units Sold", f"{highlights.get('total_units', 0):,.0f}")
    col4.metric("Avg. Order Value", format_currency(highlights.get("avg_order_value", 0.0)))

    top_city = highlights.get("top_city")
    share_online = highlights.get("online_share")

    narrative_lines = []
    if top_city:
        city_name, city_revenue = top_city
        narrative_lines.append(
            f"**Top city:** {city_name} drives {format_currency(city_revenue)} in revenue for the selected period."
        )
    if share_online is not None:
        narrative_lines.append(
            f"**Digital share:** Online channels account for {format_percent(share_online)} of revenue."
        )
    if narrative_lines:
        st.markdown("\n".join(f"- {line}" for line in narrative_lines))


available_months = list_available_months()
if not available_months:
    st.error("No sales extracts found under the data directory.")
    st.stop()

def _format_month(token: str) -> str:
    if len(token) == 6:
        return f"{token[:4]}-{token[4:]}"
    return token

default_month_selection = list(available_months)
selected_months = st.sidebar.multiselect(
    "Sales Months",
    options=available_months,
    default=default_month_selection,
    format_func=_format_month,
)

if not selected_months:
    st.sidebar.error("Select at least one month to populate the dashboard.")
    st.stop()

selected_months = tuple(sorted(selected_months))

data = load_data(selected_months)

filtered_data = data[data["Quantity_item"] >= 0].copy()
filtered_data, selections = apply_sidebar_filters(filtered_data)
selections["months_loaded"] = [_format_month(token) for token in selected_months]

transaction_level = (
    filtered_data.drop_duplicates(subset="Transaction_ID") if not filtered_data.empty else pd.DataFrame()
)
highlights = compute_highlights(filtered_data)

tabs = st.tabs(
    [
        "Sales Overview",
        "Customer Insights",
        "Transaction Value",
        "RFM Analysis",
        "Promotion Impact",
        "Retention Cohorts",
        "Data Quality",
    ]
)


with tabs[0]:
    st.header("Sales Overview")

    if filtered_data.empty:
        st.info("No data available for the selected filters.")
    else:
        render_kpi_summary(highlights)

        trend = (
            filtered_data.groupby("Date_transaction")["Quantity_item"].sum().reset_index()
        )
        if trend.empty:
            st.info("No trend data to display.")
        else:
            fig_trend = px.line(
                trend,
                x="Date_transaction",
                y="Quantity_item",
                markers=True,
                title="Sales Trend Over Time",
                color_discrete_sequence=["#0193A5"],
            )
            fig_trend.update_layout(xaxis_title="Date", yaxis_title="Units Sold")
            st.plotly_chart(fig_trend, use_container_width=True)

        monthly_sales = build_monthly_sales_series(filtered_data)
        if not monthly_sales.empty:
            monthly_sales["Month_Label"] = monthly_sales["Month"].dt.strftime("%Y-%m")
            fig_monthly = px.bar(
                monthly_sales,
                x="Month_Label",
                y="Revenue",
                hover_data={"Orders": True, "Units": True},
                title="Monthly Revenue & Volume",
                color_discrete_sequence=["#0193A5"],
            )
            fig_monthly.update_xaxes(type="category", title="Month")
            fig_monthly.update_yaxes(title="Revenue")
            fig_monthly.update_traces(
                text=monthly_sales["Orders"],
                texttemplate="Orders: %{text:,}",
                textposition="outside",
                marker_line_color="#004A59",
                marker_line_width=0.5,
            )
            if len(monthly_sales) == 1:
                st.caption("Only one month of data is available for the selected filters.")
            st.plotly_chart(fig_monthly, use_container_width=True)

        location_data = filtered_data.dropna(subset=["Latitude", "Longitude"])
        if not location_data.empty:
            location_agg = (
                location_data.groupby(["Latitude", "Longitude", "CITY"], dropna=False)["Quantity_item"].sum().reset_index()
            )
            fig_map = px.scatter_mapbox(
                location_agg,
                lat="Latitude",
                lon="Longitude",
                size="Quantity_item",
                color="Quantity_item",
                hover_name="CITY",
                hover_data={"Quantity_item": True},
                size_max=18,
                zoom=5,
                mapbox_style="carto-positron",
                color_continuous_scale=["#0193A5", "#027184", "#004A59", "#C73618", "#F16744", "#F6A278"],
                title="Sales Distribution by Location",
            )
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.info("No latitude/longitude data available for the filtered dataset.")


with tabs[1]:
    st.header("Customer Insights")

    if filtered_data.empty:
        st.info("No data available for the selected filters.")
    else:
        customer_data = filtered_data.copy()
        grouped_data = customer_data.groupby(["Contact_ID", "CITY"], dropna=False).agg(
            promotion_touches=("Promo_Flag", "sum"),
            quantity=("Quantity_item", "sum"),
            total_sales=("Line_Sales_Value", "sum"),
            total_original=("Line_Original_Value", "sum"),
        ).reset_index()

        grouped_data["Unit_price"] = grouped_data["total_sales"] / grouped_data["quantity"].replace(0, np.nan)
        grouped_data = grouped_data.replace([np.inf, -np.inf], np.nan).dropna(subset=["Unit_price"])

        if grouped_data.empty:
            st.info("Not enough customer data to build the view.")
        else:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.boxplot(grouped_data["Unit_price"], showfliers=False)
            ax.set_title("Distribution of Customer Unit Price")
            ax.set_ylabel("Unit Price")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

            quantiles = grouped_data["Unit_price"].quantile([0.25, 0.5, 0.75]).rename({0.25: "25%", 0.5: "50%", 0.75: "75%"})
            st.subheader("Customer Value Quantiles")
            st.dataframe(quantiles.to_frame(name="Unit Price"))

            boundaries = [grouped_data["Unit_price"].min(), quantiles.loc["25%"], quantiles.loc["75%"], grouped_data["Unit_price"].max()]
            labels = ["Low Value", "Medium Value", "High Value"]
            grouped_data["Customer_Category"] = pd.cut(
                grouped_data["Unit_price"],
                bins=boundaries,
                labels=labels,
                include_lowest=True,
                duplicates="drop",
            )

            grouped_data["Promotion_Used"] = np.where(grouped_data["promotion_touches"] > 0, "Used Promotion", "No Promotion")
            category_promotion_counts = pd.crosstab(grouped_data["Customer_Category"], grouped_data["Promotion_Used"])

            st.subheader("Promotion Usage by Customer Value")
            st.dataframe(category_promotion_counts)

            if category_promotion_counts.shape[1] == 2:
                chi2, p_value, dof, _ = chi2_contingency(category_promotion_counts)
                st.caption(f"Chi-squared statistic: {chi2:.2f} (p-value = {p_value:.4f}, dof = {dof})")
            else:
                st.caption("Not enough variation to run a chi-squared test on promotion usage.")

            top_customers = (
                grouped_data.sort_values("total_sales", ascending=False)
                .head(10)
                .loc[:, ["Contact_ID", "CITY", "total_sales", "quantity", "promotion_touches"]]
            )
            top_customers.rename(
                columns={
                    "total_sales": "Sales",
                    "quantity": "Units",
                    "promotion_touches": "Promotion Touches",
                },
                inplace=True,
            )
            st.subheader("Top Customers by Sales")
            st.dataframe(top_customers)


with tabs[2]:
    st.header("Transaction Value")

    if transaction_level.empty:
        st.info("No transaction data for the selected filters.")
    else:
        positive_transactions = transaction_level[transaction_level["Transaction_value"] >= 0]
        avg_transaction_value = positive_transactions["Transaction_value"].mean()
        st.metric("Average Transaction Value", format_currency(avg_transaction_value))

        bin_width = st.slider("Histogram bin width", min_value=5, max_value=100, value=20, step=5)
        data_min = float(positive_transactions["Transaction_value"].min())
        data_max = float(positive_transactions["Transaction_value"].max())
        bins = np.arange(start=data_min, stop=data_max + bin_width, step=bin_width)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(positive_transactions["Transaction_value"], bins=bins, color="#0193A5", edgecolor="#004A59", alpha=0.75)
        ax.set_title("Distribution of Transaction Values")
        ax.set_xlabel("Transaction Value")
        ax.set_ylabel("Frequency")
        ax.grid(True, axis="y", alpha=0.3)
        st.pyplot(fig)

        location_transactions = transaction_level.dropna(subset=["Latitude", "Longitude"])
        if not location_transactions.empty:
            location_avg = (
                location_transactions.groupby(["Latitude", "Longitude", "CITY"], dropna=False)["Transaction_value"].mean().reset_index()
            )
            fig_map = px.scatter_mapbox(
                location_avg,
                lat="Latitude",
                lon="Longitude",
                size="Transaction_value",
                color="Transaction_value",
                hover_name="CITY",
                size_max=18,
                mapbox_style="carto-positron",
                color_continuous_scale=["#0193A5", "#027184", "#004A59", "#C73618", "#F16744", "#F6A278"],
                title="Average Transaction Value by Location",
            )
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.info("No geo-tagged transactions available for the current filters.")


with tabs[3]:
    st.header("RFM Analysis")

    if transaction_level.empty:
        st.info("No transaction data for the selected filters.")
    else:
        rfm_data = transaction_level[["Transaction_ID", "Contact_ID", "Date_transaction", "Transaction_value", "CITY", "Latitude", "Longitude"]].dropna(subset=["Contact_ID"])

        current_date = rfm_data["Date_transaction"].max() + pd.Timedelta(days=1)
        rfm = rfm_data.groupby("Contact_ID").agg(
            Recency=("Date_transaction", lambda x: (current_date - x.max()).days),
            Frequency=("Transaction_ID", "count"),
            Monetary=("Transaction_value", "sum"),
            Latitude=("Latitude", "first"),
            Longitude=("Longitude", "first"),
            CITY=("CITY", "first"),
        )

        rfm["R_Score"] = quantile_rank_score(rfm["Recency"], reverse=True)
        rfm["F_Score"] = quantile_rank_score(rfm["Frequency"])
        rfm["M_Score"] = quantile_rank_score(rfm["Monetary"])
        rfm["RFM_Score"] = rfm[["R_Score", "F_Score", "M_Score"]].sum(axis=1)

        fig, ax = plt.subplots(figsize=(10, 5))
        rfm["RFM_Score"].hist(
            bins=range(rfm["RFM_Score"].min(), rfm["RFM_Score"].max() + 2),
            color="#0193A5",
            edgecolor="#004A59",
            align="left",
        )
        ax.set_title("Distribution of RFM Scores")
        ax.set_xlabel("RFM Score")
        ax.set_ylabel("Number of Customers")
        ax.set_xticks(range(rfm["RFM_Score"].min(), rfm["RFM_Score"].max() + 1))
        ax.grid(False)
        st.pyplot(fig)

        rfm["Segment"] = pd.cut(
            rfm["RFM_Score"],
            bins=[rfm["RFM_Score"].min() - 1, 6, 9, 12, rfm["RFM_Score"].max()],
            labels=["At Risk", "Growth", "Loyal", "Champions"],
        )
        segment_summary = rfm.groupby("Segment", observed=False).agg(
            Customers=("RFM_Score", "count"),
            Avg_R=("R_Score", "mean"),
            Avg_F=("F_Score", "mean"),
            Avg_M=("M_Score", "mean"),
            Revenue=("Monetary", "sum"),
        )
        st.subheader("RFM Segment Summary")
        st.dataframe(segment_summary)

        geo_rfm = rfm.dropna(subset=["Latitude", "Longitude"]).groupby(["Latitude", "Longitude", "CITY"], dropna=False)["RFM_Score"].mean().reset_index()
        if not geo_rfm.empty:
            fig_map = px.scatter_mapbox(
                geo_rfm,
                lat="Latitude",
                lon="Longitude",
                size="RFM_Score",
                color="RFM_Score",
                hover_name="CITY",
                size_max=18,
                zoom=5,
                mapbox_style="carto-positron",
                color_continuous_scale=["#0193A5", "#027184", "#004A59", "#C73618", "#F16744", "#F6A278"],
                title="Average RFM Score by Location",
            )
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.info("No geospatial data to display RFM scores by location.")


with tabs[4]:
    st.header("Promotion Impact")

    if filtered_data.empty:
        st.info("No data available for the selected filters.")
    else:
        promotion_summary = build_promotion_summary(filtered_data)
        if promotion_summary.empty:
            st.info("Not enough promotion data to compare performance.")
        else:
            st.subheader("Promotion vs Non-Promotion Performance")
            display_summary = promotion_summary.copy()
            display_summary["revenue"] = display_summary["revenue"].apply(format_currency)
            display_summary["avg_order_value"] = display_summary["avg_order_value"].apply(format_currency)
            display_summary["total_discount"] = display_summary["total_discount"].apply(format_currency)
            display_summary["avg_discount_per_order"] = display_summary["avg_discount_per_order"].apply(format_currency)
            st.dataframe(display_summary)

            promo_chart = promotion_summary.reset_index()
            promo_label_col = promo_chart.columns[0]
            if promo_label_col != "Promotion":
                promo_chart.rename(columns={promo_label_col: "Promotion"}, inplace=True)
            revenue_total = promo_chart["revenue"].sum()
            orders_total = promo_chart["orders"].sum()

            share_chart = promo_chart[["Promotion", "revenue", "orders"]].copy()
            if revenue_total:
                share_chart["Revenue Share"] = share_chart["revenue"] / revenue_total
            else:
                share_chart["Revenue Share"] = 0
            if orders_total:
                share_chart["Order Share"] = share_chart["orders"] / orders_total
            else:
                share_chart["Order Share"] = 0

            share_long = share_chart.melt(
                id_vars="Promotion",
                value_vars=["Revenue Share", "Order Share"],
                var_name="Metric",
                value_name="Share",
            )

            fig_share = px.bar(
                share_long,
                x="Share",
                y="Promotion",
                color="Metric",
                orientation="h",
                barmode="group",
                text="Share",
                title="Promotion Contribution to Revenue and Orders",
                color_discrete_sequence=["#004A59", "#F16744"],
            )
            fig_share.update_traces(texttemplate="%{text:.1%}", textposition="outside", textfont=dict(color="#2c3e50"))
            fig_share.update_layout(
                xaxis_tickformat=".0%",
                legend_title="",
                font=dict(color="#2c3e50"),
                bargap=0.35,
                yaxis_title="Promotion Flag",
            )
            st.plotly_chart(fig_share, use_container_width=True)

            aov_labels = promo_chart["avg_order_value"].apply(format_currency)
            fig_aov = px.bar(
                promo_chart,
                x="Promotion",
                y="avg_order_value",
                color="Promotion",
                text=aov_labels,
                title="Average Order Value by Promotion Flag",
                color_discrete_sequence=["#004A59", "#F16744"],
            )
            fig_aov.update_traces(textposition="outside", textfont=dict(color="#2c3e50"))
            fig_aov.update_layout(
                yaxis_title="Average Order Value",
                showlegend=False,
                font=dict(color="#2c3e50"),
            )
            st.plotly_chart(fig_aov, use_container_width=True)

        top_promotions = (
            filtered_data.dropna(subset=["Promotion_ID"])
            .groupby("Promotion_ID")
            .agg(
                revenue=("Transaction_value", "sum"),
                orders=("Transaction_ID", "nunique"),
                units=("Quantity_item", "sum"),
            )
            .sort_values("revenue", ascending=False)
            .head(10)
        )
        if not top_promotions.empty:
            st.subheader("Top Promotions by Revenue")
            top_promotions_table = top_promotions.copy()
            top_promotions_table["revenue"] = top_promotions_table["revenue"].apply(format_currency)
            st.dataframe(top_promotions_table)

        store_profitability = build_profitability_by_store(filtered_data).head(10)
        if not store_profitability.empty:
            st.subheader("Store Performance Snapshot")
            store_table = store_profitability.reset_index()
            store_table["revenue"] = store_table["revenue"].apply(format_currency)
            store_table["discount"] = store_table["discount"].apply(format_currency)
            st.dataframe(store_table.rename(columns={"LIBELLE_y": "Store", "CITY": "City"}))


with tabs[5]:
    st.header("Retention Cohorts")

    retention_matrix, cohort_sizes = build_cohort_retention(filtered_data)

    if retention_matrix.empty:
        st.info("Not enough sequential purchases to build a cohort matrix for the selected filters.")
    else:
        cohort_display = retention_matrix.copy()
        cohort_display = cohort_display.applymap(lambda x: f"{x:.0%}")
        st.subheader("Monthly Retention by Cohort (percentage of customers retained)")
        st.dataframe(cohort_display)

        retention_line = retention_matrix.reset_index().melt(id_vars="CohortMonth", var_name="Months Since First Purchase", value_name="Retention")
        retention_line = retention_line[retention_line["Months Since First Purchase"] <= 6]
        if not retention_line.empty:
            fig_retention = px.line(
                retention_line,
                x="Months Since First Purchase",
                y="Retention",
                color="CohortMonth",
                title="Retention Trend for First Six Months",
                markers=True,
            )
            fig_retention.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig_retention, use_container_width=True)

        st.caption("Cohort size denotes the number of distinct customers making their first purchase in that month.")
        cohort_sizes_display = cohort_sizes.to_frame(name="Customers")
        st.dataframe(cohort_sizes_display)


with tabs[6]:
    st.header("Data Quality & Notes")

    st.subheader("Filter Summary")
    summary_lines = []
    for key, values in selections.items():
        if not values:
            continue
        summary_lines.append(f"**{key.capitalize()}**: {', '.join(str(v) for v in values)}")
    if summary_lines:
        st.markdown("\n".join(f"- {line}" for line in summary_lines))
    else:
        st.markdown("- Using the full dataset (no filters applied).")

    quality_columns = [
        "Transaction_ID",
        "Contact_ID",
        "Promotion_ID",
        "Transaction_value",
        "Quantity_item",
        "Latitude",
        "Longitude",
        "CITY",
        "Catégorie",
    ]

    quality_report = generate_data_quality_report(filtered_data, quality_columns)
    if quality_report.empty:
        st.info("No data quality metrics available for the current selection.")
    else:
        report_display = quality_report.copy()
        report_display["missing_pct"] = report_display["missing_pct"].map(lambda x: f"{x:.2f}%")
        st.subheader("Missing Data Overview")
        st.dataframe(report_display)

    st.subheader("Analyst Notes")
    st.markdown(
        """
        - Promotion impact and cohort retention views rely on sufficient historical coverage. Filters that isolate a narrow window may reduce interpretability.
        - Coordinates are missing for some stores (notably online channels); these are excluded from map-based visuals.
        - Consider exporting filtered data for deeper modelling via `st.dataframe(filtered_data.head())` or downloading with Streamlit's download buttons.
        """
    )
