
import pandas as pd
import numpy as np
import joblib
from data_loader import getSymbolsData
from analyze_patterns import analyze_pattern_accuracy_full

def trainModel(symbol="DELHIVERY-EQ"):
    df = getSymbolsData(symbol)

    result_df, pattern_stats = analyze_pattern_accuracy_full(df)
    lst = joblib.dump(pattern_stats, str(symbol)+"_pattern_model.pkl")
    # print(" succesfully saved pattern_model : "+str(symbol)+"_pattern_model.pkl")
    # # print(pattern_stats)
    # print("======================================================================")
    # # print(result_df)
    # print(lst[0])
    return lst[0]
