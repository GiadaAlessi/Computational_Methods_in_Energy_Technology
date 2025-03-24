import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#### Input data ####

# Physical data
t0 = 21.0
T0 = t0 + 273.15
tw = 21.0
Tw = tw + 273.15
e = 1.0
lamb = 220
aext = 8.7
H = 1.0
W = 1.0
ro = 2400
cp = 900
A0 = 21.0
A1 = 4.3
A2 = 7.5
w1 = (2 * np.pi) / (24 * 3600)
w2 = (2 * np.pi) / (365 * 24 * 3600)

#### Numerical Solution ####

# Numerical data
N = 100
Dt = 3600 # 1 hour
Int = 100 * 24 * 3600
tol = 1e-12
B = 1  # Implicit B = 1, Explicit B = 0, Crank-Nicholson B = 0.5

# Discretization of the mesh
Dx = e / N

xcv = np.zeros(N + 1)
xP = np.zeros(N + 2)
vP = np.zeros(N + 1)

for i in range(0, N + 1):
    xcv[i] = i * Dx

xP[0] = 0
for i in range(1, N + 1):
    xP[i] = (xcv[i] + xcv[i - 1]) / 2
xP[N + 1] = e

for i in range(1, N + 1):
    vP[i] = (xcv[i] - xcv[i - 1]) * H

# Definition of Vectors
Tn = np.zeros(N + 2)
Tn1 = np.zeros(N + 2)
aw = np.zeros(N + 2)
ae = np.zeros(N + 2)
bp = np.zeros(N + 2)
ap = np.zeros(N + 2)

# Parametric Study Setup
params_to_study = {
    "thickness": [0.1, 1.0, 10.0],
    "conductivity": [1, 220, 400],
    "h_ext": [0, 8.7, 60],
}

results = {}

# Parametric studies
for param, values in params_to_study.items():
    results[param] = []
    for value in values:
        if param == "thickness":
            e = value
            Dx = e / N
            xcv = np.linspace(0, e, N + 1)
            for i in range(1, N + 1):
                xP[i] = (xcv[i] + xcv[i - 1]) / 2
            xP[N + 1] = e
        elif param == "conductivity":
            lamb = value
        elif param == "h_ext":
            aext = value

        # Initialize temperature
        Tn.fill(T0)
        times, T_x0, T_xe2, T_xe = [], [], [], []

        # Time loop
        for t in range(1, Int + 1, Dt):
            Text = A0 + A1 * np.sin(w1 * t) + A2 * np.sin(w2 * t) + 273.15
            relaxation_factor = 0.8
            converged = False
            while not converged:
                Tn1_prev = Tn1.copy()
                for i in range(1, N + 1):
                    aw[i] = lamb * H * W / (xP[i] - xP[i - 1])
                    ae[i] = lamb * H * W / (xP[i + 1] - xP[i])
                    ap[i] = ro * vP[i] * cp / Dt + aw[i] + ae[i]
                    bp[i] = ro * vP[i] * cp * Tn[i] / Dt

                aw[0] = 0
                ae[0] = lamb / (xP[1] - xP[0])
                ap[0] = ae[0] + aext
                bp[0] = aext * Text

                aw[N + 1] = 0
                ae[N + 1] = 0
                ap[N + 1] = 1
                bp[N + 1] = Tw

                # TDMA Solver
                P = np.zeros(N + 2)
                R = np.zeros(N + 2)
                P[0] = ae[0] / ap[0]
                R[0] = bp[0] / ap[0]
                for i in range(1, N + 2):
                    P[i] = ae[i] / (ap[i] - aw[i] * P[i - 1])
                    R[i] = (bp[i] + aw[i] * R[i - 1]) / (ap[i] - aw[i] * P[i - 1])

                Tn1[N + 1] = R[N + 1]
                for i in range(N, -1, -1):
                    Tn1[i] = P[i] * Tn1[i + 1] + R[i]

                Tn1 = relaxation_factor * Tn1 + (1 - relaxation_factor) * Tn
                max_diff = np.max(np.abs(Tn1 - Tn1_prev))
                if max_diff < tol:
                    converged = True

            Tn[:] = Tn1[:]

            times.append(t / (3600 * 24))
            T_x0.append(Tn[0] - 273.15)
            T_xe2.append(Tn[N // 2] - 273.15)
            T_xe.append(Tn[N + 1] - 273.15)

        results[param].append({
            "value": value,
            "T_x0": T_x0[-1],
            "T_xe2": T_xe2[-1],
            "T_xe": T_xe[-1],
        })

# Save the parametric results to a CSV file
parametric_data = []
for param, studies in results.items():
    for study in studies:
        parametric_data.append({
            "Parameter": param,
            "Value": study["value"],
            "Final Temp (x=0) [°C]": study["T_x0"],
            "Final Temp (x=e/2) [°C]": study["T_xe2"],
            "Final Temp (x=e) [°C]": study["T_xe"],
        })

parametric_df = pd.DataFrame(parametric_data)
parametric_df.to_csv("parametric_study_results.csv", index=False)

print("Parametric study results saved to 'parametric_study_results.csv'.")
