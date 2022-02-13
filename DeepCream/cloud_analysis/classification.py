import numpy as np
import pandas as pd

from DeepCream.cloud_analysis.analysis import Analysis


class Classification:
    def __init__(self, analysis: Analysis):
        pass

    def get_classification(self) -> pd.DataFrame:
        # Pandas DataFrame with columns for each cloud type (the first two
        # columns determine the center of the cloud in ratio to the image
        # dimensions), rows for each cloud and
        # values between 0 and 1 for the probabilities
        pass
