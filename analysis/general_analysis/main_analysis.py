from core.config.analysis_config import AnalysisConfig
from core.runners.analysis_runner import run_analysis_from_config


def main():
    config = AnalysisConfig()
    run_analysis_from_config(config)


if __name__ == "__main__":
    main()