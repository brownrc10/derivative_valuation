import os
from pathlib import Path
from dotenv import load_dotenv
from src.stock_data import StockData
from src.barrier_option import BarrierOption
from src.montecarlo import BarrierMonteCarloSimulation

load_dotenv()
BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")


def main():
    stock_data_file_path = Path("parquet_output/ticker_data/five_minute_stock_data")
    stock_data = StockData(
        stock_file_path=stock_data_file_path, valuation_date="2024-04-01"
    )
    summarized_stock_data = stock_data.summarize(BASE_URL=BASE_URL, API_KEY=API_KEY)


if __name__ == "__main__":
    main()
