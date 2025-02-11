
# Genetic Algorithms and Genetic Programming Project

This project implements various approaches and versions of Genetic Algorithms (GA) and Genetic Programming (GP) to solve optimization and robot simulation problems. Each file in the project represents a different stage in the development and evaluation of these techniques.

## File Descriptions

1. **`RobotV2_1_GA_FinalAprox.py`**  
   - **Description:** Implements the final version of the Genetic Algorithm (GA) in approach 2.1.  
   - **Purpose:** Contains advanced optimizations and a robust solution refined through multiple iterations.  

2. **`RobotV2_0_GA_NonEvolution.py`**  
   - **Description:** A variant of the Genetic Algorithm that does not utilize active evolution.  
   - **Purpose:** Serves as a baseline to evaluate how results are affected when evolutionary mechanisms are not applied.  

3. **`Version3_0_GP_FirstAprox.py`**  
   - **Description:** The first approximation of Genetic Programming (GP) in version 3.  
   - **Purpose:** Acts as a starting point for GP implementation, but no evolution is observed in this version.  
   - **Note:** This file is designed to analyze initial behavior without significant evolution.  

4. **`Version3_1_GP_parallelized_SecondAprox.py`**  
   - **Description:** The second approximation of Genetic Programming (GP) in version 3, now with parallelization.  
   - **Purpose:** Introduces parallel processing to improve efficiency, but as in version `3_0`, no evolution is observed in the results.  
   - **Note:** This version is useful for evaluating how parallelism affects performance in scenarios without evolution.  

5. **`Version3_2GP_FinalAprox.py`**  
   - **Description:** Final version of the 3.2 approximation of the Genetic Programming model.  
   - **Purpose:** Implements key optimizations and resolves issues observed in previous versions, showing a functional and evolved model.  

## Notes on Evolution

- **Versions without evolution:**  
  `Version3_0_GP_FirstAprox.py` and `Version3_1_GP_parallelized_SecondAprox.py` are designed to study and diagnose cases where evolution does not occur. These versions are key to understanding limiting factors and improving the algorithms.  

- **Optimized final versions:**  
  `RobotV2_1_GA_FinalAprox.py` and `Version3_2GP_FinalAprox.py` are examples where issues have been resolved, and evolved results are observed.


## Author

Created by [Arahí Fernández Monagas](https://github.com/afm719).  

