import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np
from vnstock3 import Vnstock
from scipy.optimize import minimize

# Đọc file CSV (thay đường dẫn tới file CSV của bạn)
df = pd.read_csv('E:/D/DC/std/tailieu/Ki7/Code/PTDLTC/result3.csv')

# Tạo đối tượng Vnstock (để lấy dữ liệu giá cổ phiếu)
def create_vnstock_instance():
    return Vnstock().stock(symbol='VN30F1M', source='VCI')

# Lấy dữ liệu giá cổ phiếu
def fetch_stock_data(tickers, start_date, end_date):
    data = pd.DataFrame()
    skipped_tickers = []  # Danh sách các mã bị bỏ qua do không có dữ liệu
    for ticker in tickers:
        stock = Vnstock().stock(symbol=ticker, source='VCI')
        try:
            stock_data = stock.quote.history(start=str(start_date), end=str(end_date))
            if stock_data is not None and not stock_data.empty:
                stock_data = stock_data[['time', 'close']]
                stock_data.columns = ['time', ticker]
                stock_data.set_index('time', inplace=True)
                if data.empty:
                    data = stock_data
                else:
                    data = data.join(stock_data, how='inner')  # Join để tránh mã cổ phiếu không có dữ liệu
            else:
                skipped_tickers.append(ticker)
        except IndexError:
            skipped_tickers.append(ticker)

    return data, skipped_tickers

# Vẽ biểu đồ giá cổ phiếu
def plot_stock_chart(data, tickers):
    available_tickers = [ticker for ticker in tickers if ticker in data.columns]

    if not available_tickers:
        st.warning("Không có mã cổ phiếu nào có dữ liệu để hiển thị.")
        return

    data_reset = data.reset_index()  # Reset index để giữ cột 'time'
    data_long = pd.melt(data_reset, id_vars=['time'], value_vars=available_tickers, 
                        var_name='Mã cổ phiếu', value_name='Giá đóng cửa')

    fig = px.line(data_frame=data_long, x='time', y='Giá đóng cửa', color='Mã cổ phiếu',
                  title="Giá đóng cửa của các mã cổ phiếu")
    
    fig.update_layout(xaxis_title='Ngày', yaxis_title='Giá đóng cửa (VND)')
    st.plotly_chart(fig)

# Tính toán hiệu suất danh mục đầu tư
def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(weights * mean_returns)
    std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = returns / std_dev  # Tỷ lệ Sharpe
    return std_dev, returns, sharpe_ratio

# Tối ưu hóa danh mục đầu tư theo mô hình Markowitz
def optimize_portfolio(mean_returns, cov_matrix, risk_tolerance):
    num_assets = len(mean_returns)
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})  # Tổng trọng số = 1
    bounds = tuple((0, 1) for asset in range(num_assets))  # Trọng số trong khoảng từ 0 đến 1

    # Hàm tối thiểu hóa rủi ro với mức khẩu vị rủi ro của người dùng
    def objective_function(weights):
        std_dev, returns, _ = portfolio_performance(weights, mean_returns, cov_matrix)
        return std_dev * (1 - risk_tolerance) - returns * risk_tolerance

    result = minimize(objective_function,
                      num_assets * [1. / num_assets],  # Trọng số ban đầu đều nhau
                      method='SLSQP',
                      bounds=bounds,
                      constraints=constraints)

    return result.x

# Vẽ biểu đồ đường biên hiệu quả (Efficient Frontier) với thông tin tỷ lệ cổ phiếu
def plot_efficient_frontier(mean_returns, cov_matrix, tickers, num_portfolios=10000):
    results = np.zeros((3, num_portfolios))
    weights_record = []
    weight_str_list = []  # Danh sách chứa chuỗi tỷ lệ cổ phiếu

    for i in range(num_portfolios):
        weights = np.random.random(len(mean_returns))
        weights /= np.sum(weights)
        portfolio_std_dev, portfolio_return, sharpe_ratio = portfolio_performance(weights, mean_returns, cov_matrix)
        results[0, i] = portfolio_std_dev
        results[1, i] = portfolio_return
        results[2, i] = sharpe_ratio  # Tỷ lệ Sharpe
        weights_record.append(weights)

        # Chuyển đổi tỷ lệ thành chuỗi để hiển thị khi hover
        weight_str = ", ".join([f"{ticker}: {weight * 100:.2f}%" for ticker, weight in zip(tickers, weights)])
        weight_str_list.append(weight_str)

    # Vẽ đường biên hiệu quả với thông tin tỷ lệ cổ phiếu hiển thị khi hover
    fig = px.scatter(x=results[0, :], y=results[1, :], color=results[2, :],
                     hover_data={'Tỷ lệ cổ phiếu': weight_str_list},
                     labels={'x': 'Rủi ro (Độ lệch chuẩn)', 'y': 'Lợi nhuận kỳ vọng', 'color': 'Tỷ lệ Sharpe'},
                     title="Đường biên hiệu quả Markowitz (Efficient Frontier)")
    st.plotly_chart(fig)

# Hiển thị danh sách mã cổ phiếu đã chọn và xử lý thao tác xóa
def display_selected_stocks(df):
    if st.session_state.selected_stocks:
        st.markdown("### Danh sách mã cổ phiếu đã chọn:")
        for stock in st.session_state.selected_stocks:
            # Tìm thông tin chi tiết từ DataFrame dựa trên symbol
            stock_info = df[df['symbol'] == stock]
            if not stock_info.empty:
                organ_name = stock_info.iloc[0]['organ_name']
                icb_name = stock_info.iloc[0]['icb_name']
                exchange = stock_info.iloc[0]['exchange']

                # Hiển thị thông tin: Mã cổ phiếu, tên công ty, và ngành
                col1, col2, col3, col4, col5 = st.columns([2, 4, 3, 2, 1])  # Thêm cột để hiển thị thông tin
                col1.write(stock)  # Mã cổ phiếu
                col2.write(organ_name)  # Tên công ty
                col3.write(icb_name)  # Tên ngành
                col4.write(exchange)
                if col4.button(f"❌", key=f"remove_{stock}"):  # Nút xóa
                    st.session_state.selected_stocks.remove(stock)
                    st.rerun()  # Làm mới lại giao diện sau khi xóa
    else:
        st.write("Chưa có mã cổ phiếu nào được chọn.")

# Tạo session state để lưu mã cổ phiếu đã chọn
if 'selected_stocks' not in st.session_state:
    st.session_state.selected_stocks = []

# Giao diện người dùng để lọc từ file CSV
st.title("Dashboard Hỗ trợ Xây dựng Danh mục Đầu tư Chứng khoán")

# Bộ lọc theo sàn giao dịch (exchange)
selected_exchange = st.selectbox('Chọn sàn giao dịch', df['exchange'].unique())

# Lọc dữ liệu dựa trên sàn giao dịch đã chọn
filtered_df = df[df['exchange'] == selected_exchange]

# Bộ lọc theo loại ngành (icb_name)
selected_icb_name = st.selectbox('Chọn ngành', filtered_df['icb_name'].unique())

# Lọc dữ liệu dựa trên ngành đã chọn
filtered_df = filtered_df[filtered_df['icb_name'] == selected_icb_name]

# Bộ lọc theo mã chứng khoán (symbol)
selected_symbols = st.multiselect('Chọn mã chứng khoán', filtered_df['symbol'])

# Lưu các mã chứng khoán đã chọn vào session state khi nhấn nút "Thêm mã"
if st.button("Thêm mã vào danh sách"):
    for symbol in selected_symbols:
        if symbol not in st.session_state.selected_stocks:
            st.session_state.selected_stocks.append(symbol)
    st.success(f"Đã thêm {len(selected_symbols)} mã cổ phiếu vào danh mục!")

# Hiển thị danh sách mã cổ phiếu đã chọn và xử lý thao tác xóa
display_selected_stocks(df)

# Lựa chọn thời gian lấy dữ liệu
start_date = st.date_input("Ngày bắt đầu", value=pd.to_datetime("2020-01-01"))
end_date = st.date_input("Ngày kết thúc", value=pd.to_datetime("2023-01-01"))

# Lựa chọn khẩu vị rủi ro
risk_tolerance = st.slider("Chọn mức khẩu vị rủi ro (0: Rủi ro thấp, 1: Rủi ro cao)", 0.0, 1.0, 0.5)

# Lấy dữ liệu giá cổ phiếu và hiển thị biểu đồ
if st.button("Lấy dữ liệu"):
    tickers = st.session_state.selected_stocks
    if tickers:
        data, skipped_tickers = fetch_stock_data(tickers, start_date, end_date)
        if data.empty:
            st.warning("Không có đủ dữ liệu cho các mã cổ phiếu đã chọn.")
        else:
            plot_stock_chart(data, tickers)

            # Thông báo nếu có mã cổ phiếu bị bỏ qua
            if skipped_tickers:
                st.warning(f"Các mã sau không có dữ liệu trong khoảng thời gian được chọn: {', '.join(skipped_tickers)}")

            # Tính toán tỷ suất sinh lợi và ma trận hiệp phương sai
            returns = data.pct_change().dropna()  # Tỷ suất sinh lợi hàng ngày
            mean_returns = returns.mean()
            cov_matrix = returns.cov()

            # Hiển thị ma trận hiệp phương sai
            st.subheader("Ma trận hiệp phương sai:")
            st.dataframe(cov_matrix)

            # Tối ưu hóa danh mục đầu tư
            if len(mean_returns) == len(tickers):
                optimized_weights = optimize_portfolio(mean_returns, cov_matrix, risk_tolerance)

                # Tính toán và hiển thị các chỉ số của danh mục đầu tư tối ưu
                std_dev, expected_return, sharpe_ratio = portfolio_performance(optimized_weights, mean_returns, cov_matrix)
                st.subheader("Danh mục đầu tư tối ưu:")
                for ticker, weight in zip(tickers, optimized_weights):
                    st.write(f"{ticker}: {weight * 100:.2f}%")

                st.write(f"Lợi nhuận kỳ vọng: {expected_return * 100:.2f}%")
                st.write(f"Rủi ro (Độ lệch chuẩn): {std_dev * 100:.2f}%")
                st.write(f"Tỷ lệ Sharpe: {sharpe_ratio:.2f}")

                # Vẽ đường biên hiệu quả (Efficient Frontier) và hiển thị tỷ lệ cổ phiếu
                plot_efficient_frontier(mean_returns, cov_matrix, tickers)
            else:
                st.warning("Không đủ dữ liệu để tối ưu hóa danh mục đầu tư.")
    else:
        st.warning("Vui lòng chọn ít nhất một mã cổ phiếu để lấy dữ liệu.")
