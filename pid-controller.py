## https://www.freecodecamp.org/news/basic-control-theory-with-python/
import matplotlib.pyplot as plt
import control as ctrl

num = [10]  ## Set the system gain to 10
den = [2, 2, 1]  ## Defines the denominator. G(s) = 10 / (2 s^2 + 2 s + 1).

G = ctrl.TransferFunction(num, den)

Kp = 5
Ki = 2
Kd = 1

C = ctrl.TransferFunction([Kd, Kp, Ki], [1, 0])

CL = ctrl.feedback(C * G, 1)

plt.figure(figsize=(10, 6))
ctrl.root_locus(C * G, grid=True)
plt.title("Root Locus Plot (Closed-Loop)")

# Step 6: Plot Bode Plot for Closed-Loop System
plt.figure(figsize=(10, 6))
ctrl.bode_plot(CL, dB=True, Hz=False, deg=True)
plt.suptitle("Bode Plot (Closed-Loop)", fontsize=16)

# Step 7: Plot Nyquist Plot for Closed-Loop System
plt.figure(figsize=(10, 6))
ctrl.nyquist_plot(CL)
plt.title("Nyquist Plot (Closed-Loop)")

plt.show()
