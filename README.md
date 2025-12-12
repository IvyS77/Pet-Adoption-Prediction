## Overview

This project investigates the relationship between pet characteristics and adoption outcomes using data from  
PetFinder.my, Malaysia's largest pet adoption platform.  

I focused on several characteristics such as:  
- Species  
- Breed  
- Age  
- Color  
- Gender  
- Health  
- Number of photos *(one of the strongest predictors of adoption speed)*  

Using these features, I explored the data, created visualizations, and trained machine learning models to predict adoption speed (0–4).  
The goal is to help identify which pets may need more attention earlier, so shelters can better allocate resources and reduce stress and euthanasia for animals.

## How to Run the Code

1. Place `final.py` inside the same folder as `train.csv`.
2. Open Terminal and navigate to the folder:
   ```bash
   cd path/to/your/project/train

3. Run the script:

   ```bash
   python3 final.py
   ```
4. The following output files will be generated:

   * `age_histogram.png`
   * `state_pet_counts.png`
   * `confusion_matrix_rf.png`
   * `feature_importance_rf.png`

Model accuracy will also be printed in the terminal.

## Results Summary

* Random Forest achieved **~34% accuracy**
* Most important predictors:

  * Breed
  * Age
  * Number of photos

State distribution shows high concentration in PetFinder state codes **41326** and **41401**
(Malaysia regions, not U.S. states).