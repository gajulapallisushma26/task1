from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
square_footage = [1500, 2000, 2500, 1800, 2200, 1600, 1900, 2100, 2300, 2600]
bedrooms = [3, 4, 3, 4, 3, 2, 3, 4, 4, 4]
bathrooms = [2, 2.5, 3, 2, 2.5, 1.5, 2, 2.5, 3, 3.5]
prices = [300000, 400000, 500000, 350000, 450000, 320000, 380000, 420000, 460000, 520000]
X=list(zip(square_footage,bedrooms,bathrooms))
Y=prices
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, Y_train)
predictions=model.predict(X_test)
mse=mean_squared_error(Y_test,predictions)
print("mean squared error:",mse)
new_house=[[1800,3,2]]
predicted_price=model.predict(new_house)
print("predicted price for the new house:",predicted_price)

