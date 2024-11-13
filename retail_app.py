import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import matplotlib.dates as mdates
import plotly.express as px
from scipy.stats import chi2_contingency
import numpy as np

@st.cache_data
def load_data():
    # 读取经纬度数据并合并
    # geo_data_path = '/Users/yuanhaowen/BB project/qc_cities_coordinates.csv'
    # geo_data = pd.read_csv(geo_data_path)
    geo_data_url = "https://retailproject2024.blob.core.windows.net/retaildata/qc_cities_coordinates.csv?sv=2022-11-02&ss=bfqt&srt=co&sp=rtfx&se=2025-11-13T10:49:04Z&st=2024-11-13T02:49:04Z&spr=https&sig=T8Y0dAHVRz7h6lyFuCVGp3%2FixKM8KTKy%2FDV9XxIz9bY%3D"
    geo_data = pd.read_csv(geo_data_url)
    # 读取原始数据
    # data_path = '/Users/yuanhaowen/BB project/combined_data_202301.xlsx'
    # data = pd.read_excel(data_path)
    data_url = "https://retailproject2024.blob.core.windows.net/retaildata/combined_data_202301.csv?sv=2022-11-02&ss=bfqt&srt=co&sp=rtfx&se=2025-11-13T10:49:04Z&st=2024-11-13T02:49:04Z&spr=https&sig=T8Y0dAHVRz7h6lyFuCVGp3%2FixKM8KTKy%2FDV9XxIz9bY%3D"
    data = pd.read_excel(data_url)

    # 处理日期和数据格式
    data['Date_transaction'] = pd.to_datetime(data['Date_transaction'], format='%Y%m%d')
    data['Year'] = data['Date_transaction'].dt.year
    data['Quarter'] = data['Date_transaction'].dt.quarter
    data['Month'] = data['Date_transaction'].dt.month
    data['Unit_sale_price'] = data['Unit_sale_price'].astype(str).str.replace(',', '.').astype(float)
    data['Item_ID'] = data['Item_ID'].astype(str)  # 确保 Item_ID 是字符串类型
    data['days_since_epoch'] = (data['Date_transaction'] - pd.Timestamp('1970-01-01')).dt.days


    # 将原始数据和经纬度数据合并
    merged_data = data.merge(geo_data, left_on='CITY', right_on='City', how='left')
    merged_data.loc[merged_data['LIBELLE_y'].isin(['0902 Ecom Website - LVER Canada', '0904 Ecom Website - LVER USA']), 'CITY'] = 'Online'

    return merged_data


data = load_data()
# 初始化 filtered_data 为 data 的副本
filtered_data = data.copy()
filtered_data = filtered_data[filtered_data['Quantity_item'] >= 0]


tabs = st.tabs(["Sales Visualization", "Customer Analysis", "Transaction Value", "RFM Analysis"])

with tabs[0]:
    # 筛选器
    selected_years = st.sidebar.multiselect('Select Years:', options=sorted(data['Year'].unique().astype(str)))
    if selected_years:
        filtered_data = filtered_data[filtered_data['Year'].isin([int(year) for year in selected_years])]

    selected_quarters = st.sidebar.multiselect('Select Quarters:', options=sorted(filtered_data['Quarter'].unique().astype(str)))
    if selected_quarters:
        filtered_data = filtered_data[filtered_data['Quarter'].isin([int(quarter) for quarter in selected_quarters])]

    selected_months = st.sidebar.multiselect('Select Months:', options=sorted(filtered_data['Month'].unique().astype(str)))
    if selected_months:
        filtered_data = filtered_data[filtered_data['Month'].isin([int(month) for month in selected_months])]

    # 日期范围选择器
    min_day = filtered_data['days_since_epoch'].min()
    max_day = filtered_data['days_since_epoch'].max()
    selected_days = st.sidebar.slider('Select Date Range:', min_value=min_day, max_value=max_day, value=(min_day, max_day))
    # 显示选择的日期范围
    start_date = pd.Timestamp('1970-01-01') + pd.to_timedelta(selected_days[0], unit='D')
    end_date = pd.Timestamp('1970-01-01') + pd.to_timedelta(selected_days[1], unit='D')
    st.sidebar.write(f"Selected Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    filtered_data = filtered_data[(filtered_data['days_since_epoch'] >= selected_days[0]) & (filtered_data['days_since_epoch'] <= selected_days[1])]


    # 类别筛选器
    selected_categories = st.sidebar.multiselect('Select Categories:', options=sorted(filtered_data['Catégorie'].dropna().unique()))
    if selected_categories:
        filtered_data = filtered_data[filtered_data['Catégorie'].isin(selected_categories)]

    # 商品ID筛选器
    selected_item_ids = st.sidebar.multiselect('Select Item IDs:', options=sorted(filtered_data['Item_ID'].unique()))
    if selected_item_ids:
        filtered_data = filtered_data[filtered_data['Item_ID'].isin(selected_item_ids)]

    # 城市筛选器
    selected_cities = st.sidebar.multiselect('Select Cities:', options=sorted(filtered_data['CITY'].unique()))
    if selected_cities:
        filtered_data = filtered_data[filtered_data['CITY'].isin(selected_cities)]

    # 门店名称筛选器，基于选择的城市动态更新选项
    if selected_cities:
        options_stores = sorted(filtered_data[filtered_data['CITY'].isin(selected_cities)]['LIBELLE_y'].unique())
    else:
        options_stores = sorted(filtered_data['LIBELLE_y'].unique())
    selected_stores = st.sidebar.multiselect('Select Stores:', options=options_stores)
    if selected_stores:
        filtered_data = filtered_data[filtered_data['LIBELLE_y'].isin(selected_stores)]

    # 检查过滤后的数据是否为空
    if filtered_data.empty:
        st.write("No data available for the selected filters.")
    else:
        # 绘制图表
        grouped = filtered_data.groupby('Date_transaction')['Quantity_item'].sum().reset_index()
        fig, ax = plt.subplots(figsize=(10, 5))
        # 绘制数据点和线
        ax.plot(grouped['Date_transaction'], grouped['Quantity_item'], marker='o', linestyle='-', color='tab:blue')
        # 设置标题和轴标签
        ax.set_title('Sales Trend Over Time', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Total Quantity Sold', fontsize=12)
        # 设置 Y 轴的标签为整数
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
        # 添加网格线
        #ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        # 自动旋转日期标记以防止重叠
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        # 显示图表
        st.pyplot(fig)


        # filtered_data = filtered_data[filtered_data['CITY'] != 'Online']
        # Aggregate sales data by location
        location_data = filtered_data.groupby(['Latitude', 'Longitude', 'CITY'])['Quantity_item'].sum().reset_index()

        # Create a map using Plotly
        fig_map = px.scatter_mapbox(
            location_data,
            lat='Latitude',
            lon='Longitude',
            size='Quantity_item',
            color='Quantity_item',
            hover_name='CITY',
            hover_data={
                'CITY': False,
                'Quantity_item': True,
                'Latitude': False,
                'Longitude': False
            },
            size_max=15,
            zoom=6,
            mapbox_style='carto-positron',
    #        color_continuous_scale= "Icefire"
            color_continuous_scale=["#0193A5", "#027184", "#004A59", "#C73618", "#F16744", "#F6A278"]
        )

        fig_map.update_layout(
            title='Sales Distribution by Location',
            mapbox_center={
                'lat': location_data['Latitude'].mean(),
                'lon': location_data['Longitude'].mean()
            },
            margin={'l': 0, 'r': 0, 't': 30, 'b': 0}
        )

        # Display the map in Streamlit
        st.plotly_chart(fig_map)



# filtered_data = data.copy()
# filtered_data = filtered_data[filtered_data['Quantity_item'] >= 0]

with tabs[1]:
    # 对 'Unit_original_price' 进行转换
    filtered_data['Unit_original_price'] = filtered_data['Unit_original_price'].astype(str).str.replace(',', '.').astype(float)
    filtered_data['Promotion_ID'] = filtered_data['Promotion_ID'].apply(lambda x: 0 if pd.isna(x) else 1)
    
    # 计算 'Total_Original_Price'
    filtered_data = filtered_data.assign(Total_Original_Price=filtered_data['Unit_original_price'] * filtered_data['Quantity_item'])

    # 按 'Contact_ID', 'CITY' 分组并汇总数据
    grouped_data = filtered_data.groupby(['Contact_ID', 'CITY']).agg({
        'Promotion_ID': 'sum',
        'Quantity_item': 'sum',
        'Total_Original_Price': 'sum'
    }).reset_index()

    # 计算 'Unit_price'
    grouped_data['Unit_price'] = grouped_data['Total_Original_Price'] / grouped_data['Quantity_item']


    # 绘制盒须图
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot(grouped_data['Unit_price'], showfliers=False)  # 不显示离群点
    ax.set_title('Boxplot of Unit Price')
    ax.set_ylabel('Unit Price')
    ax.set_ylim(0, 100)  # 设置y轴的上限为100
    ax.grid(True)  # 添加网格线
    st.pyplot(fig)

    quantiles = grouped_data['Unit_price'].quantile([0.25, 0.50, 0.75])
    st.write("Quantiles:", quantiles)

    # 设置分割点以包含所有数据
    boundaries = [grouped_data['Unit_price'].min(), quantiles[0.25], quantiles[0.75], grouped_data['Unit_price'].max()]
    labels = ['Low Value Customer', 'Medium Value Customer', 'High Value Customer']

    # 使用 cut 函数创建新的分类列
    grouped_data['Customer_Category'] = pd.cut(grouped_data['Unit_price'], bins=boundaries, labels=labels, include_lowest=True)

    # 统计每个客户类别下的 Promotion_ID 出现频次
    category_promotion_counts = grouped_data.groupby(['Customer_Category']).agg({
        'Promotion_ID': ['sum', 'count']
    })
    category_promotion_counts.columns = ['Used Promotion', 'No Promotion']

    # 在 Streamlit 中显示统计数据
    st.table(category_promotion_counts)

    # 执行卡方检验
    chi2, p_value, dof, expected = chi2_contingency([category_promotion_counts['Used Promotion'], category_promotion_counts['No Promotion']])

    # 在 Streamlit 中打印结果
    st.write("Chi-squared Test Statistic:", chi2)
    st.write("p-value:", p_value)
    st.write("Degrees of freedom:", dof)
    # st.write("Expected frequencies:", expected)

    # 在 Streamlit 中显示前几行数据
    st.write("Grouped Data:", filtered_data.head())


with tabs[2]:
    # 对 'Transaction_value' 进行数据转换
    # 检查过滤后的数据是否为空
    if filtered_data.empty:
        st.write("No data available for the selected filters.")
    else:
        # 对 'Transaction_value' 进行数据转换
        filtered_data['Transaction_value'] = filtered_data['Transaction_value'].astype(str).str.replace(',', '.').astype(float)

        # 去除重复的交易 ID 并筛选出大于等于0的交易值
        unique_transaction_values = filtered_data.drop_duplicates(subset='Transaction_ID', keep='first')
        unique_transaction_values = unique_transaction_values[unique_transaction_values['Transaction_value'] >= 0]

        # 定义bin的宽度和X轴的最大边界
        bin_width = 20
        x_max = 400

        # 计算bin的边界
        data_min = unique_transaction_values['Transaction_value'].min()
        # 确保bin的最大边界不超过设定的x_max
        bin_edges = np.arange(start=data_min, stop=min(unique_transaction_values['Transaction_value'].max(), x_max) + bin_width, step=bin_width)

        # 创建直方图
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(unique_transaction_values['Transaction_value'], bins=bin_edges, color='#0193A5', edgecolor='#004A59', alpha=0.7)

        # 设置标题和轴标签
        ax.set_title('Distribution of Transaction Values')
        ax.set_xlabel('Transaction Value')
        ax.set_ylabel('Frequency')

        # 设置X轴刻度，使用bin的边界
        ax.set_xticks(bin_edges)
        ax.set_xticklabels([f'{edge:.0f}' for edge in bin_edges], rotation=90)  # 旋转标签以更好地显示

        # 设置X轴的显示范围到400
        ax.set_xlim([data_min, x_max])

        # 在 Streamlit 中显示图形
        st.pyplot(fig)


    # 检查过滤后的数据是否为空
    if filtered_data.empty:
        st.write("No data available for the selected filters.")
    else:
        # filtered_data = filtered_data[filtered_data['CITY'] != 'Online']
        # Aggregate sales data by location
        location_data = unique_transaction_values.groupby(['Latitude', 'Longitude', 'CITY'])['Transaction_value'].mean().reset_index()

        # Create a map using Plotly
        fig_map = px.scatter_mapbox(
            location_data,
            lat='Latitude',
            lon='Longitude',
            size='Transaction_value',
            color='Transaction_value',
            hover_name='CITY',
            hover_data={
                'CITY': False,
                'Transaction_value': True,
                'Latitude': False,
                'Longitude': False
            },
            size_max=15,
            zoom=6,
            mapbox_style='carto-positron',
    #        color_continuous_scale= "Icefire"
            color_continuous_scale=["#0193A5", "#027184", "#004A59", "#C73618", "#F16744", "#F6A278"]
        )

        fig_map.update_layout(
            title='Transaction Value by Location',
            mapbox_center={
                'lat': location_data['Latitude'].mean(),
                'lon': location_data['Longitude'].mean()
            },
            margin={'l': 0, 'r': 0, 't': 30, 'b': 0}
        )

        # Display the map in Streamlit
        st.plotly_chart(fig_map)






with tabs[3]:

    # 检查过滤后的数据是否为空
    if filtered_data.empty:
        st.write("No data available for the selected filters.")
    else:
        # 提取 RFM 相关列
        rfm_data = unique_transaction_values[['Transaction_ID', 'Contact_ID', 'Date_transaction', 'Transaction_value', 'Latitude', 'Longitude', 'CITY']]

        # 计算 RFM 指标
        current_date = rfm_data['Date_transaction'].max() + pd.Timedelta(days=1)
        rfm = rfm_data.groupby('Contact_ID').agg({
            'Date_transaction': lambda x: (current_date - x.max()).days,  # Recency: Days since last purchase
            'Transaction_ID': 'count',  # Frequency: Number of transactions
            'Transaction_value': 'sum',  # Monetary: Total money spent
            'Latitude': 'first',  # 取每个 Contact_ID 的第一条记录的 Latitude
            'Longitude': 'first',  # 取每个 Contact_ID 的第一条记录的 Longitude
            'CITY': 'first'  # 取每个 Contact_ID 的第一条记录的 CITY
        }).rename(columns={
            'Date_transaction': 'Recency',
            'Transaction_ID': 'Frequency',
            'Transaction_value': 'Monetary'
        })

        # 给 Recency, Frequency, 和 Monetary 分配分数
        rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1], duplicates='drop')  # 更近的购买得分更高
        rfm['F_Score'] = pd.cut(rfm['Frequency'], bins=[0, 1, 2, 4, 7, rfm['Frequency'].max()], labels=[1, 2, 3, 4, 5], right=True)
        rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1, 2, 3, 4, 5], duplicates='drop')  # 金钱花费更高得分也更高

        # 重新计算总 RFM 得分
        rfm['RFM_Score'] = rfm['R_Score'].astype(int) + rfm['F_Score'].astype(int) + rfm['M_Score'].astype(int)

        # 在 Streamlit 中显示 RFM 数据的头几行，去除 Latitude 和 Longitude 列
        st.write("RFM Table:", rfm.drop(columns=['Latitude', 'Longitude']))

        # 绘制 RFM 分数的分布直方图
        fig, ax = plt.subplots(figsize=(10, 6))
        rfm['RFM_Score'].hist(bins=range(rfm['RFM_Score'].min(), rfm['RFM_Score'].max() + 2), color='#0193A5', edgecolor='#004A59', align='left')
        ax.set_title('Distribution of RFM Scores')
        ax.set_xlabel('RFM Score')
        ax.set_ylabel('Number of Customers')
        ax.set_xticks(range(rfm['RFM_Score'].min(), rfm['RFM_Score'].max() + 1))  # 设置 X 轴的刻度，显示每个得分
        ax.grid(False)
        st.pyplot(fig)


    # 检查过滤后的数据是否为空
    if filtered_data.empty:
        st.write("No data available for the selected filters.")
    else:
        # filtered_data = filtered_data[filtered_data['CITY'] != 'Online']
        # Aggregate sales data by location
        location_data = rfm[rfm['CITY'] != 'Online']
        location_data = location_data.groupby(['Latitude', 'Longitude', 'CITY'])['RFM_Score'].mean().reset_index()

        # Create a map using Plotly
        fig_map = px.scatter_mapbox(
            location_data,
            lat='Latitude',
            lon='Longitude',
            size='RFM_Score',
            color='RFM_Score',
            hover_name='CITY',
            hover_data={
                'CITY': False,
                'RFM_Score': True,
                'Latitude': False,
                'Longitude': False
            },
            size_max=15,
            zoom=6,
            mapbox_style='carto-positron',
    #        color_continuous_scale= "Icefire"
            color_continuous_scale=["#0193A5", "#027184", "#004A59", "#C73618", "#F16744", "#F6A278"]
        )

        fig_map.update_layout(
            title='Transaction Value by Location',
            mapbox_center={
                'lat': location_data['Latitude'].mean(),
                'lon': location_data['Longitude'].mean()
            },
            margin={'l': 0, 'r': 0, 't': 30, 'b': 0}
        )

        # Display the map in Streamlit
        st.plotly_chart(fig_map)
