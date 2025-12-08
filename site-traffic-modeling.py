import re
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("datasets/nasa_site_traffic.csv", parse_dates=["minute"])

df["lambda_per_sec"] = df["count"] / 60

print(df.head())


total_requests = df["count"].sum()
total_minutes = len(df)

lambda_global = total_requests / (total_minutes * 60)
print(f"Global arrival rate: {lambda_global:.2f} requests per second")


# plt.plot(df["minute"], df["lambda_per_sec"])
# plt.xlabel("Time")
# plt.ylabel("λ (requests per second)")
# plt.title("Arrival Rate over Time")
# plt.show()


window = 60

df["mean"] = df["count"].rolling(window=window).mean()
df["std"] = df["count"].rolling(window=window).std()

df["zscore"] = (df["count"] - df["mean"]) / df["std"]  ## How many standard deviations from the mean is the count?

df["is_burst"] = df["zscore"].abs() > 3


## FOr a Poisson distribution, the variance is sqrt(lambda). We approximate the burst if count is more than 3 sigma from the mean.
df["burst_poisson"] = (df["count"] > (df["mean"] + 3 * np.sqrt(df["mean"]))) | (
    df["count"] < (df["mean"] - 3 * np.sqrt(df["mean"]))
)

print(df.head())
print(df[(df["burst_poisson"] != df["is_burst"])])


burstiness_index = df["count"].var() / df["count"].mean()
print(f"Burstiness index: {burstiness_index:.2f}")


plt.figure(figsize=(16, 5))
plt.plot(df["minute"], df["count"], label="Traffic")
plt.scatter(df[df["is_burst"]]["minute"], df[df["is_burst"]]["count"], label="Bursts", marker="o")
plt.legend()
plt.title("Burst Detection in NASA Traffic")
plt.show()
