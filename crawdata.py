import pandas as pd
import schedule
import time
from vnstock3 import Vnstock

# Hàm chính để chạy công việc lấy dữ liệu và lưu vào file CSV
def run_task():
    # Tạo đối tượng Vnstock
    stock = Vnstock().stock(symbol='VN30F1M', source='VCI')

    # Lấy danh sách các ngành và mã chứng khoán theo ICB
    stock_icb = stock.listing.symbols_by_industries()[['symbol', 'organ_name', 'icb_code1']]

    # Lấy dữ liệu các ngành ICB
    icb_all = stock.listing.industries_icb()[['icb_name','icb_code']]

    # Tạo bảng result từ các thông tin lấy được
    result = pd.merge(stock_icb, icb_all, left_on='icb_code1', right_on='icb_code', how='inner')[['icb_name']]

    result2 = pd.merge(stock_icb, icb_all, left_on='icb_code1', right_on='icb_code', how='inner')[['symbol', 'organ_name', 'icb_name']]

    # Lấy tất cả các mã chứng khoán từ các sàn HOSE, HNX, UPCOM, VN30, HNX30
    def get_stocks_by_exchange(stock, exchange):
        symbols = stock.listing.symbols_by_group(exchange).tolist()
        return pd.DataFrame({"symbol": symbols, "exchange": exchange})

    def get_all_stocks(stock):
        exchanges = ['HOSE', 'HNX', 'UPCOM', 'VN30', 'HNX30']
        stock_dfs = [get_stocks_by_exchange(stock, exchange) for exchange in exchanges]
        return pd.concat(stock_dfs, ignore_index=True)

    # Tạo bảng kết hợp với các mã chứng khoán từ các sàn
    all_stocks_df = get_all_stocks(stock)
    
    # Ghép bảng result2 với all_stocks_df
    result3 = pd.merge(result2, all_stocks_df, on='symbol', how='inner')

    # Lưu kết quả vào file CSV
    result3.to_csv('result3.csv', index=False)

    print("Công việc đã hoàn thành và dữ liệu được cập nhật!")

# Thiết lập lịch chạy hàng ngày vào lúc 14:00
schedule.every().day.at("00:00").do(run_task)

# Vòng lặp chính để kiểm tra công việc và thực hiện nếu đến giờ
while True:
    schedule.run_pending()
    time.sleep(1)
