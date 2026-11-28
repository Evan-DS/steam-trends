"""What Makes a Steam Game Successful? ECE 9603/9063 course project.

The package is organised like this:
    config.py        folders, random seed and the settings shared by every step
    data.py          reading the source files and the files the steps make
    modeling.py      the models, cross-validation and scoring used by the modelling steps
    plotting.py      the chart style shared by every figure
    printing.py      neat tables in the terminal
    progress.py      timestamped status messages
    runlog.py        saves everything printed to a log file
    scrub.py         the --scrub command (removes everything the program generated)
    kaggle_client.py  Kaggle access, only used when regenerating the source data
    pipeline.py      runs the steps in order (python main.py)
    steps/           one module per step of the analysis
"""

__version__ = "1.0.0"
