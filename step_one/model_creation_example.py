import pandas as pd, numpy as np, seaborn as sns, matplotlib.pyplot as plt


df = pd.read_csv('Salary_Data.csv')

df.head()

x = df.YearsExperience.values
y = df.Salary.values

from sklearn.preprocessing import Normalizer

from sklearn.linear_model import LinearRegression


model = LinearRegression()
model.fit(x.reshape(-1,1), y.reshape(-1,1))

prediction = model.predict(np.array(3).reshape(-1,1))

number = prediction.flatten()[0]

predictions_input = model.predict(x.reshape(-1,1))


from sklearn.externals import joblib
joblib.dump(model, "linear_regression_model.pkl")

# from sklearn.externals import joblib
# filename = "linear_regression_model.pkl"
# joblib.dump(model, open(filename, 'wb'))

