import numpy as np
import matplotlib.pyplot as plt

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
Tn1g=np.zeros(N + 2)
aw = np.zeros(N + 2)
ae = np.zeros(N + 2)
bp = np.zeros(N + 2)
ap = np.zeros(N + 2)
times = []
T_x0 = []
T_xe2 = []
T_xe = []
q_in_list = []
q_out_list = []

# Initial Map t=0
for i in range(N + 2):
    Tn[i] = T0

# Initial Conditions
T_x0.append(T0 - 273.15)
T_xe2.append(T0 - 273.15)
T_xe.append(T0 - 273.15)
times.append(0)
q_in_list.append(0)
q_out_list.append(0)

# Evaluation of Temperature at Any Instant
for t in range(1, Int + 1, Dt):
    Text = A0 + A1 * np.sin(w1 * t) + A2 * np.sin(w2 * t) + 273.15
    relaxation_factor = 0.8 
    converged = False
    while not converged:
        Tn1g[:] = Tn1[:]
        # Coefficients for internal nodes
        for i in range(1, N + 1):
            aw[i] = lamb * H * W / (xP[i] - xP[i - 1])
            ae[i] = lamb * H * W / (xP[i + 1] - xP[i])
            ap[i] = ro * vP[i] * cp / Dt + aw[i] + ae[i]
            bp[i] = (ro * vP[i] * cp * Tn[i] / Dt) + (1 - B) * (aw[i] * Tn[i - 1] - (aw[i] + ae[i]) * Tn[i] + ae[i] * Tn[i + 1])

        # Boundary conditions
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

        # Update temperature for next time step
        Tn1[:] = relaxation_factor * Tn1 + (1 - relaxation_factor) * Tn
        # Convergence Check
        max_diff = np.max(np.abs(Tn1 - Tn1g))
        if max_diff < tol:
            converged = True
    
    # Final Update
    Tn[:] = Tn1[:]

    # Store results for plotting
    times.append(t / (3600 * 24))  # Convert time to days
    T_x0.append(Tn[0] - 273.15)
    T_xe2.append(Tn[N // 2] - 273.15)
    T_xe.append(Tn[N + 1] - 273.15)
    
    # Calculate heat fluxes
    q_in = -lamb * H * W * (Tn[1] - Tn[0]) / (xP[1] - xP[0])
    q_out = -lamb * H * W * (Tn[N + 1] - Tn[N]) / (xP[N + 1] - xP[N])
    q_in_list.append(q_in)
    q_out_list.append(q_out)



# External T over time
Text_array = [A0 + A1 * np.sin(w1 * t) + A2 * np.sin(w2 * t) for t in range(0, 3600*24*365 + 1, Dt)]
times_for_Text = [t / (3600*24) for t in range(0, 3600*24*365 + 1, Dt)]


# Plot Heat Fluxes
plt.figure()
plt.plot(times, q_in_list, label="Heat Flux at x=0 (q_in)", color='red')
plt.plot(times, q_out_list, label="Heat Flux at x=e (q_out)", color='green')
plt.xlabel("Time (Days)")
plt.ylabel("Heat Flux (W)")
plt.title("Heat Flux at the Boundaries Over Time")
plt.legend()
plt.grid(True)
plt.show()

# Initialize global energy balance array
global_EB = np.zeros(N + 2)

# Energy balance for each control volume
for i in range(1, N + 1):
    global_EB[i] = (aw[i] * Tn[i - 1] + ae[i] * Tn[i + 1] + bp[i]) - (ap[i] * Tn[i])

# Boundary conditions energy balance
global_EB[0] = (ae[0] * Tn[1] + bp[0]) - (ap[0] * Tn[0])  # x=0
global_EB[N + 1] = bp[N + 1] - (ap[N + 1] * Tn[N + 1])  # x=e

# Sum up of the global energy balance
total_EB = np.sum(global_EB)

# Output and check the global energy balance
print(f"Global Energy Balance: {total_EB:.6f}")
if abs(total_EB) > 1:
    print(f"WARNING: Energy balance not satisfied. Total EB: {total_EB:.6f}")


# Plot results
plt.figure()
plt.plot(times, T_x0, label="T at x=0 (°C)")
plt.plot(times, T_xe2, label="T at x=e/2 (°C)")
plt.plot(times, T_xe, label="T at x=e (°C)")
plt.xlabel("Time (Days)")
plt.ylabel("Temperature (°C)")
plt.title("Temperature Evolution")
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(times_for_Text, np.array(Text_array), label="T_ext (°C)", linestyle="-")
plt.xlabel("Time")
plt.ylabel("Temperature (°C)")
plt.title("External Temperature Evolution in a Year")
plt.legend()
plt.grid(True)
plt.show()
