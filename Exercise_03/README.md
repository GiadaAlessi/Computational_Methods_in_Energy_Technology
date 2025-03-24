# Exercise 03 - Numerical Solution for 1D Steady Heat Conduction in a Tube with Forced and Natural Convection

This project models the **one-dimensional steady-state heat conduction** in a cylindrical tube subjected to both **internal forced convection** and **external natural convection**. The solution is achieved using the **finite volume method** combined with the **Gauss-Seidel** and **TDMA** algorithms.

## 📄 Report
For a detailed explanation of the methodology, results, and conclusions, please refer to the full report (developed in collaboration with **Emanuele Kob**):

➡️ [Conjugated Problem Report - Giada Alessi & Emanuele Kob](https://github.com/GiadaAlessi/Computational_Methods_in_Energy_Technology/blob/main/Exercise_03/Alessi-Kob_CMET_Assign-3.pdf)

## 🔍 Project Overview
The project explores:
- Numerical solution of steady-state conduction using the **finite volume method**.
- Evaluation of internal forced convection and external natural convection effects.
- Iterative process using:
  - **Gauss-Seidel algorithm** for solving internal fluid properties.
  - **TDMA (Tri-Diagonal Matrix Algorithm)** for solving the conduction problem in the tube thickness.
- Analysis of thermophysical properties for different fluids:
  - **Water**, **Therminol 66**, and **Air**.

## 🟧🟦 MATLAB Code
The MATLAB code implements:
- Discretization of the geometry using finite volumes.
- Iterative calculations to achieve convergence on velocity, pressure, and temperature profiles.
- Detailed evaluation of heat transfer coefficients and their effect on system performance.

For code details, refer to the file ➡️ [Conjugated Problem Code - Giada Alessi & Emanuele Kob](https://github.com/GiadaAlessi/Computational_Methods_in_Energy_Technology/blob/main/Exercise_03/Alessi-Kob_CMET_Assign-3_MATLAB.pdf)

---
**Authors:** Giada Alessi & Emanuele Kob
