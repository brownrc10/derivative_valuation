import os
from dotenv import load_dotenv
from src.stock_data import StockData
from src.barrier_option import BarrierOption
from src.montecarlo import BarrierMonteCarloSimulation

load_dotenv()
BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")


def main():
    pass


if __name__ == "__main__":

    main()
