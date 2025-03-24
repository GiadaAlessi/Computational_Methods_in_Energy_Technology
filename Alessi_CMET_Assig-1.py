# Simulation of a steady, 1D conduction heat transfer in a cylindrical wall with internal heat production

import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, Eq, ln, solve

#### Input data ####

# Physical data
ta=150
Ta=ta+273.15
tb=25
Tb=tb+273.15
Ra=0.05
Rb=0.1
H=2
lambda_=45
alfaa=600
alfab=20
qv=1e5

# Numerical data
N=10
tol=10e-12
tstart=150
Tstart = tstart + 273.15

#### Analytical Solution ####

# Definition of variables
r = symbols('r')
C1, C2 = symbols('C1 C2')

# Solution in cylindrical coordinates
T = -qv / (4 * lambda_) * r**2 + C1 * ln(r) + C2

# Derivative of temperature
dT_dr = T.diff(r)

# Boundary conditions
# R=Ra
eq1 = Eq(-lambda_ * dT_dr.subs(r, Ra), alfaa * (Ta - T.subs(r, Ra)))
# R=Rb
eq2 = Eq(-lambda_ * dT_dr.subs(r, Rb), alfab * (T.subs(r, Rb) - Tb))

# Find solution
solutions = solve((eq1, eq2), (C1, C2))
T_sol = T.subs(solutions)

# Definition of the position of the nodes
Dr_a=(Rb-Ra)/N

rcv_a=np.zeros(N+1)
rP_a=np.zeros(N+2)

for i in range (0, N+1, 1):
    rcv_a[i]=Ra+(i)*Dr_a

rP_a[0]=Ra
for i in range (1, N+1, 1):
    rP_a[i] = (rcv_a[i]+rcv_a[i-1])/2
rP_a[N+1]=Rb


T_a = np.zeros(N + 2)
def T_sol_numeric(r_value):
    return T_sol.subs(r, r_value).evalf()

# Temperature at each point
for i in range(0, N + 2, 1):
    T_a[i] = T_sol_numeric(rP_a[i]) - 273.15

# Temperature distribution Analytical
print("Analytical Temperature Distribution:")
for i in range(0, N + 2, 1):
    print(f"At r = {rP_a[i]:.4f} T: {T_a[i]:.6f} °C")

# Plotting the temperature profile Analytical
plt.plot(rP_a, T_a, marker='o')
plt.xlabel('Radial Position [m]')
plt.ylabel('Temperature [°C]')
plt.title('Analytical Temperature Profile')
plt.grid()
plt.show()

#### Numerical Solution ####

# Discretization of the mesh
Dr=(Rb-Ra)/N

rcv=np.zeros(N+1)
rP=np.zeros(N+2)
vP=np.zeros(N+1)

for i in range (0, N+1, 1):
    rcv[i]=Ra+(i)*Dr

rP[0]=Ra
for i in range (1, N+1, 1):
    rP[i] = (rcv[i]+rcv[i-1])/2
rP[N+1]=Rb

for i in range (1, N+1, 1):
    vP[i]= np.pi*(rcv[i]**2-rcv[i-1]**2)*H

# Initial temperature guess
Tg=np.zeros(N+2)
for i in range (0, N+2, 1):
    Tg[i]=Tstart

# Discretization of coefficients
aw = np.zeros (N + 2)
ae = np.zeros (N + 2)
bp = np.zeros (N + 2)
ap = np.zeros (N + 2)

# Internal nodes
for i in range (1, N+1, 1):
    aw[i] = lambda_*2*np.pi*rcv[i-1]*H/(rP[i]-rP[i-1])
    ae[i] = lambda_*2*np.pi*rcv[i]*H/(rP[i+1]-rP[i])
    ap[i] = aw[i] + ae[i]
    bp[i] = qv*vP[i]

# Boundary nodes
aw[0] = 0
ae[0] = lambda_/(rP[1]-rP[0])
ap[0] = ae[0] + alfaa
bp[0] = alfaa*Ta

aw[N+1] = lambda_/(rP[N+1]-rP[N])
ae[N+1] = 0
ap[N+1] = aw[N+1] + alfab
bp[N+1] = alfab*Tb

#### Resolution with Gauss-Seidel ####

T_ng = np.zeros(N + 2) 
convergence = tol + 1
iteration = 0 

while convergence > tol:
    convergence = 0 
    for i in range(0, N + 2, 1):
        if i == 0:  # Boundary condition at Ra
            T_ng[i] = (ae[i] * Tg[i + 1] + bp[i]) / ap[i]
        elif i == N + 1:  # Boundary condition at Rb
            T_ng[i] = (aw[i] * T_ng[i - 1] + bp[i]) / ap[i]
        else:  # Internal nodes
            T_ng[i] = (ae[i] * Tg[i + 1] + aw[i] * T_ng[i - 1] + bp[i]) / ap[i]
        
        # Convergence check
        convergence = max(convergence, abs(T_ng[i] - Tg[i]))

    # Update the old temperature
    Tg[:] = T_ng[:]

    iteration += 1 

# Temperature distribution Gauss-Seidel
print ('Gauss-Seidel temperature distribution:')
for i in range(0, N + 2, 1):
    print(f"At r = {rP[i]:.4f} T: {T_ng[i]-273.15:.6f} °C")
print(f'Converged after {iteration} iterations.')

# Plotting the temperature profile Gauss-Seidel
plt.plot(rP, T_ng - 273.15, marker='o')
plt.xlabel('Radial Position [m]')
plt.ylabel('Temperature [°C]')
plt.title('Gauss-Seidel Temperature Profile')
plt.grid()
plt.show()

#### Resolution with TDMA ####

T_nt = np.zeros (N+2)
P = np.zeros (N+2)
R = np.zeros (N+2)

# Evaluation of the triangular matrix
for i in range(0, N + 2, 1):
    if i == 0:
        P[i] = ae[i] / ap[i]
        R[i] = bp[i] / ap[i]
    else:
        P[i] = ae[i] / (ap[i] - aw[i] * P[i - 1])
        R[i] = (bp[i] + aw[i] * R[i - 1]) / (ap[i] - aw[i] * P[i - 1])

# Temperature of each node
T_nt[N+1] = R[N+1]
for i in range(N, -1, -1):
    T_nt[i] = P[i] * T_nt[i + 1] + R[i]

# Temperature distribution TDMA
print('TDMA temperature distribution:')
for i in range(0, N + 2):
    print(f"At r = {rP[i]:.4f} T: {T_nt[i] - 273.15:.6f} °C")

# Plotting the temperature profile TDMA
plt.plot(rP, T_nt - 273.15, marker='o')
plt.xlabel('Radial Position [m]')
plt.ylabel('Temperature [°C]')
plt.title('TDMA Temperature Profile')
plt.grid()
plt.show()

# Plotting the temperature profiles for both analytical and numericals solutions
plt.figure(figsize=(10, 6))
plt.plot(rP_a, T_a, marker='o', label='Analytical Solution', color='blue')
plt.plot(rP, T_ng - 273.15, marker='x', label='Gauss-Seidel', color='red') 
plt.plot(rP, T_nt - 273.15, marker='s', label='TDMA', color='green') 

plt.xlabel('Radial Position [m]')
plt.ylabel('Temperature [°C]')
plt.title('Temperature Profile Comparison')
plt.grid()
plt.legend() 
plt.show()

#### Global Balances ####

#Analytical
Qg_a=qv*np.pi*H*(Rb**2-Ra**2)
Qin_a=alfaa*2*np.pi*Ra*H*(Ta-T_a[0])
Qout_a=alfab*2*np.pi*Rb*H*(T_a[N+1]-Tb)
print (Qg_a+Qin_a ,'=', Qout_a)

#Gauss-Seidel
Qg_ng=qv*np.pi*H*(Rb**2-Ra**2)
Qin_ng=alfaa*2*np.pi*Ra*H*(Ta-T_ng[0])
Qout_ng=alfab*2*np.pi*Rb*H*(T_ng[N+1]-Tb)
print (Qg_ng+Qin_ng ,'=', Qout_ng)

#TDMA
Qg_nt=qv*np.pi*H*(Rb**2-Ra**2)
Qin_nt=alfaa*2*np.pi*Ra*H*(Ta-T_nt[0])
Qout_nt=alfab*2*np.pi*Rb*H*(T_nt[N+1]-Tb)
print (Qg_nt+Qin_nt ,'=', Qout_nt)

# Pointwise errors comparison with the analytical solution
# Calculate ppm errors
pointwise_errors_ng_ppm = (np.abs(T_a - (T_ng - 273.15)) / np.abs(T_a)) * 1e6  # Gauss-Seidel
pointwise_errors_nt_ppm = (np.abs(T_a - (T_nt - 273.15)) / np.abs(T_a)) * 1e6  # TDMA

# Maximum ppm errors for both methods
max_error_ng_ppm = np.max(pointwise_errors_ng_ppm)
max_error_nt_ppm = np.max(pointwise_errors_nt_ppm)

# Print maximum errors
print(f'\nMaximum Gauss-Seidel Error: {max_error_ng_ppm:.6f} ppm')
print(f'Maximum TDMA Error: {max_error_nt_ppm:.6f} ppm')