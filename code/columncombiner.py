import pandas as pd
from sklearn.tree import DecisionTreeRegressor

df = pd.read_excel('C:\\Users\\Kabir\\Downloads\\MLprojdata.xlsx')
X1 = df['Point_A_X']
Y1 = df['Point_A_Y']
X2 = df['Point_B_X']
Y2 = df['Point_B_Y']
X3 = df['Point_C_X']
Y3 = df['Point_C_Y']
X4 = df['Point_D_X']
Y4 = df['Point_D_Y']

df['Coordinates_A'] = df.apply(lambda row: f"({row['Point_A_X']}, {row['Point_A_Y']})", axis=1)
df['Coordinates_B'] = df.apply(lambda row: f"({row['Point_B_X']}, {row['Point_B_Y']})", axis=1)
df['Coordinates_C'] = df.apply(lambda row: f"({row['Point_C_X']}, {row['Point_C_Y']})", axis=1)
df['Coordinates_D'] = df.apply(lambda row: f"({row['Point_D_X']}, {row['Point_D_Y']})", axis=1)

df2 = df.drop(columns=['Point_A_X','Point_A_Y','Point_B_X','Point_B_Y','Point_C_X','Point_C_Y','Point_D_X','Point_D_Y'])
print(df2)
df2.head()
df.to_excel('MLprojDSdata2.xlsx', index=False)

