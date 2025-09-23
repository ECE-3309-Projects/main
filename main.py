#Saxon Scarborough R#11832063
#Jakob Salinas     R#11831816
import pandas as pd
import sklearn as skl
from sklearn.metrics import r2_score, mean_squared_error

sheet = pd.read_excel('uber_fare_sample.xlsx') #main file
sheet_copy = sheet.copy() #copy for testing row 254 in excel

#used for dropping certain columns from the data
price = "fare (USD)"
not_used = ["Trip ID","drop off time", "pick up location", "drop off location", "fare (USD)"]

#drops row 254 and the extra title column from the testing set
sheet = sheet.drop(index=250)
sheet = sheet.drop(index=251)
sheet = sheet.drop(index=252)

#converts time
sheet["pickup_hour"] = pd.to_datetime(sheet["pick up time"], errors="coerce").dt.hour
not_used = not_used + ["pick up time"]

#sets days to be integer numbers
if sheet["day"].dtype == "O":
    sheet["day"] = sheet["day"].astype("category").cat.codes

#sets the inputs and output based on dropped columns
weighted_inputs = sheet.drop(columns=not_used, errors='ignore')
output = sheet[price]

#simple data train with 40% training size on a linear regression model
in_train, in_test, out_train, out_test = skl.model_selection.train_test_split(weighted_inputs, output, test_size=0.4)
model = skl.linear_model.LinearRegression()
model.fit(in_train, out_train)
y_pred = model.predict(in_test)

print("Accuracy:", model.score(in_test, out_test))
print("weights of each column in order:", model.coef_)
print("Feature order:", list(in_train.columns))
print("R^2 = ",r2_score(out_test, y_pred))
print("Mean squared error:", mean_squared_error(out_test, y_pred))

#includes only row 254 for prediction
ids = 252
x = sheet_copy.iloc[[ids]].copy()

#duplicate of data fixing seen above
x["pickup_hour"] = pd.to_datetime(x["pick up time"], errors="coerce").dt.hour
x = x.drop(columns=not_used, errors="ignore")
if x["day"].dtype == "O":
    x["day"] = x["day"].astype("category").cat.codes

#sets x as data input for the one row
x = x.reindex(columns=in_train.columns)
x = x.fillna(in_train.median(numeric_only=True))

#prediction command to estimate the output fare price
prediction = float(model.predict(x)[0])

print(f"Predicted fare for row 254: ${prediction:.2f}") #.2f ensures two digits after the decimal
