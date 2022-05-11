import numpy as np
import pandas as pd

### DATA EXPORT
# a = [1,2,3,4]
# df = pd.DataFrame(a)
# df.to_csv("imsi.csv", header=False, index=False)


# 1
# 2
# 3
# 4


### DATA IMPORT
# df = pd.read_csv("imsi.csv", index=0)
# print(df)


### ==========================================================================

### EXPORT : array to csv
# arr = np.array([[1,2,3],[3,4,5],[4,5,6]])
arr = np.zeros(9).reshape((3,3))
df = pd.DataFrame(arr)
df.to_csv("imsi2.csv", index=False)


# 0	 1	2   < header line. 읽을 때는 무시하고 읽으므로 괜찮음.
# 1	 2	3
# 3	 4	5
# 4	 5	6


### IMPORT : csv to array
df = pd.read_csv("imsi2.csv")
df = np.array(df)
print(df)
