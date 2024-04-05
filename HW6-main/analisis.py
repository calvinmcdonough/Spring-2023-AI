from typing import Hashable
import numpy as np
import pandas as pd 

def main():
    train = pd.read_csv("train.csv")
 

    count = train['income'].value_counts()[1]
    print((count/7000)*100 , "%")

    
    


main()