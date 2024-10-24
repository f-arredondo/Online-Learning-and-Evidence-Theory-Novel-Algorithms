# Overview of Algorithms

## 1. Evidence Halving (EH)
**Description:**  
The EH class implements a learning algorithm with multiple `SGDClassifier` experts and a perfect expert. Initially, it assigns equal mass to each expert and makes aggregated predictions based on the experts' outputs. If the coarsened prediction results in an abstention, the experts' masses are updated according to their prediction accuracy using the Dempster-Shafer rule (a refined mass of 1 is assigned to the set of experts who were correct, while those who were incorrect receive mass of 0).

**Key Features:**
- **Perfect Expert Integration:** Includes a perfect expert whose predictions are always accurate, aiding in generating true labels.
- **Coarsened Predictions:** Utilizes coarsening to determine the consensus prediction among experts, handling cases of abstention effectively.
- **Dynamic Mass Updates:** Adjusts the masses of experts based on their prediction performance using the Dempster-Shafer rule.
- **Adversarial Robustness:** Designed to minimize errors in adversarial scenarios by leveraging expert feedback for mass updates.

---

## 2. Evidence Soft Halving (ESH)
**Description:**  
The ESH class implements a learning algorithm with multiple `SGDClassifier` experts. Initially, it assigns equal mass to each expert and makes aggregated predictions based on their outputs. If the coarsened prediction results in an error, the experts' masses are updated according to their prediction accuracy using the Shafer-Dempster rule. A refined mass of \( \alpha > 0 \) is assigned to the set of experts who were correct, while those who were incorrect receive a mass of \( 1 - \alpha \). No updates are made when all experts agree, as this situation is considered uninformative. The goal is to aggregate predictions in adversarial settings and minimize errors through dynamic mass updates.

**Key Features:**
- **Coarsened Predictions:** Aggregated predictions are derived from the experts’ outputs, ensuring a collective decision-making process.
- **Dynamic Mass Updates:** Experts' masses are adjusted based on their performance, promoting more reliable experts over time, using the Dempster-Shafer rule.
- **Adversarial Robustness:** Designed to function effectively in adversarial environments, aiming to minimize misclassifications through careful mass allocation.
- **Non-informative Scenarios:** Automatically skips mass updates when all experts agree, avoiding unnecessary adjustments in consensus situations.

---

## 3. Evidence Weighted Majority (EWM)
**Description:**  
The EWM class implements a learning algorithm with multiple `SGDClassifier` experts. Initially, it assigns equal mass to each expert and makes aggregated predictions based on their outputs. If the coarsened prediction results in an abstention or an error, the masses of incorrect experts are reduced by a factor of \( \beta > 0 \), and the residual mass is allocated to the remaining experts to ensure that the total mass sums to 1. No updates are made when all experts agree, as this situation is considered uninformative. The goal is to aggregate predictions in adversarial settings and minimize errors through dynamic mass updates.

**Key Features:**
- **Coarsened Predictions:** Aggregated predictions are derived from the outputs of the experts, enabling a collective decision-making process.
- **Dynamic Mass Adjustments:** Experts’ masses are updated based on their performance, allowing for more reliable experts to maintain higher masses over time.
- **Error Reduction Mechanism:** The use of a scaling factor \( \beta \) for updating masses provides a controlled method to decrease the influence of incorrect predictions while allocating the mass to the set of  remaining experts.
- **Adversarial Robustness:** Specifically designed to operate effectively in adversarial environments, minimizing errors through strategic mass allocations.
- **Informativeness Check:** Automatically skips mass updates when all experts agree, avoiding unnecessary adjustments in consensus scenarios.

---

## 4. Evidence Weighted Majority with Normalization (EWMWN)
**Description:**  
The EWMWN class implements a learning algorithm with multiple `SGDClassifier` experts. Initially, it assigns equal mass to each expert. It makes aggregated predictions based on the experts' outputs. If the coarsened prediction results in an error, the masses of incorrect experts are reduced by a factor of \( \beta \), and then all the experts' masses are normalized to maintain the sum of the masses at 1. No updates are made when all experts agree, as the situation is considered uninformative. The goal is to aggregate predictions in adversarial settings and minimize errors through dynamic mass updates.

**Key Features:**
- **Coarsened Predictions:** Aggregated predictions are derived from the outputs of the experts, enabling a collective decision-making process.
- **Dynamic Mass Adjustments:** Experts’ masses are updated based on their performance, allowing for more reliable experts to maintain higher masses over time.
- **Error Reduction Mechanism:** The use of a scaling factor \( \beta \) for updating masses provides a controlled method to decrease the influence of incorrect predictions.
- **Normalization of Masses:** After adjusting the masses of incorrect experts, normalization ensures that the total mass remains equal to 1, preserving the integrity of the mass function.
- **Adversarial Robustness:** Specifically designed to operate effectively in adversarial environments, minimizing errors through strategic mass allocations.
- **Informativeness Check:** Automatically skips mass updates when all experts agree, avoiding unnecessary adjustments in consensus scenarios.

---

## 5. Evidence Consistent (EC)
**Description:**  
The Evidence Consistent Algorithm (EC) is a robust learning framework designed to leverage multiple `SGDClassifier` experts alongside a perfect expert. It focuses on dynamically adjusting the mass allocations among subsets of experts based on their prediction accuracy in adversarial environments. The algorithm aims to minimize prediction errors while effectively aggregating expert outputs.

**Key Features:**
- **Perfect Expert Integration:** Includes a perfect expert whose predictions are always accurate, aiding in generating true labels.
- **Coarsened Predictions:** Aggregated predictions are derived from the outputs of the experts, enabling a collective decision-making process.
- **Dynamic Mass Allocation:** Initializes a mass function with a total mass of 1 for the entire set of experts. Masses are updated according to expert predictions and accuracy using the Dempster-Shafer rule, allowing for adaptive responses to varying prediction scenarios.
- **Adversarial Robustness:** Specifically designed to operate effectively in adversarial environments, minimizing errors through strategic mass allocations.


---

## 6. Evidence Soft Consistent (ESC)
**Description:**  
The Evidence Soft Consistent Algorithm (ESC) leverages multiple `SGDClassifier` experts to dynamically aggregate predictions in adversarial settings. The algorithm begins with an initial mass of 1 assigned to the entire set of experts. It updates the masses of expert subsets based on their prediction accuracy, utilizing the Shafer-Dempster rule to refine the masses. When the coarsened prediction results in an abstention or error, the masses for the subsets are adjusted accordingly. If all experts agree, no updates are made, as this situation does not provide new information. The overarching goal is to minimize errors through adaptive mass updates.

**Key Features:**
- **Coarsened Predictions:** Aggregated predictions are derived from the outputs of the experts, enabling a collective decision-making process.
- **Dynamic Mass Allocation:** Initializes a mass function with a total mass of 1 for the entire set of experts. Maintains and updates a mass function representing the confidence in subsets of experts using Dempster-Shafer rule, allowing for flexible responses to prediction outcomes.
- **Adversarial Robustness:** Designed to operate effectively in challenging, adversarial environments by generating adversarial labels and aggregating predictions accordingly.
- **Informativeness Check:** Automatically skips mass updates when all experts agree, avoiding unnecessary adjustments in consensus scenarios.

## Testing the Algorithms
Each algorithm can be tested on a synthetic dataset created using make_classification from the scikit-learn library. The dataset can be split into training and test sets using train_test_split. 

## Installation

This project requires Python version 3.10.12 and the following libraries:

- **NumPy** (version >= 1.22.0)
- **Scikit-learn** (version >= 0.24)
- **itertools** (part of the Python standard library, no installation needed)
- **Matplotlib** (version >= 3.4.0, for visualizations)

