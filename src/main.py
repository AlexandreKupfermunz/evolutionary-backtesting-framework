from core.config.backtest_config import BacktestConfig
from core.runners.backtest_runner import run_backtest_from_config


def main():
    config = BacktestConfig()
    run_backtest_from_config(config)


if __name__ == "__main__":
    main()