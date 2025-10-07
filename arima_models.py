import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf


np.random.seed(42)
n = 120
time = pd.date_range(start="2015-01-01", periods=n, freq="M")

trend = 0.5 * np.arange(n)
noise = np.random.normal(scale=2, size=n)
series = 50 + trend + noise
ts = pd.Series(series, index=time)


ts.plot(title="Synthetic Monthly Time series", figsize=(10, 6))
plt.show()


model1 = ARIMA(ts, order=(1, 1, 1)).fit()
model2 = ARIMA(ts, order=(2, 1, 2)).fit()

print("\n Model Camparison:")
print("ARIMA(1,1,1): σ² = {:.3f}, AIC = {:.2f}, BIC = {:.2f}".format(model1.scale, model1.aic, model1.bic))
print("ARIMA(2,1,2): σ² = {:.3f}, AIC = {:.2f}, BIC = {:.2f}".format(model2.scale, model2.aic, model2.bic))

# 4) Residual plots for diagnostics
fig, axes = plt.subplots(2, 2, figsize=(12, 6))

axes[0, 0].plot(model1.resid)
axes[0, 0].set_title("Residuals ARIMA(1,1,1)")

plot_acf(model1.resid.dropna(), ax=axes[0, 1], lags=30)
axes[0, 1].set_title("ACF Residuals ARIMA(1,1,1)")

axes[1, 0].plot(model2.resid)
axes[1, 0].set_title("Residuals ARIMA(2,1,2)")

plot_acf(model2.resid.dropna(), ax=axes[1, 1], lags=30)
axes[1, 1].set_title("ACF Residuals ARIMA(2,1,2)")

plt.tight_layout()
plt.show()
