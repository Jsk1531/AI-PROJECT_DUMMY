
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score 
import numpy as np 

def evaluate_model(y_test, y_pred):                                              
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"MAE  : {mae:,.3f}")
    print(f"MSE  : {mse:,.3f}")
    print(f"RMSE : {rmse:,.3f}")
    print(f"R²   : {r2:.3f}")
    return mae, mse, rmse, r2