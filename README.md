# Overview of Algorithms

## 1. Evidence Halving (EH)
**Description:**  
The EH class implements a learning algorithm with multiple `SGDClassifier` experts and a perfect expert. Initially, it assigns equal mass to each expert and makes aggregated predictions based on the experts' outputs. If the coarsened prediction results in an abstention, the experts' masses are updated according to their prediction accuracy using the Dempster-Shafer rule (a refined mass of 1 is assigned to the set of experts who were correct, while those who were incorrect receive mass of 0).

**Key Features:**
- **Dynamic Mass Updates:** Adjusts the masses of experts based on their prediction performance.
- **Perfect Expert Integration:** Includes a perfect expert whose predictions are always accurate, aiding in generating true labels.
- **Coarsened Predictions:** Utilizes coarsening to determine the consensus prediction among experts, handling cases of abstention effectively.
- **Adversarial Robustness:** Designed to minimize errors in adversarial scenarios by leveraging expert feedback for mass updates.

---

## 2. Evidence Soft Halving (ESH)
**Description:**  
The ESH class implements a learning algorithm with multiple `SGDClassifier` experts. Initially, it assigns equal mass to each expert and makes aggregated predictions based on their outputs. If the coarsened prediction results in an error, the experts' masses are updated according to their prediction accuracy using the Shafer-Dempster rule. A refined mass of \( \alpha > 0 \) is assigned to the set of experts who were correct, while those who were incorrect receive a mass of \( 1 - \alpha \). No updates are made when all experts agree, as this situation is considered uninformative. The goal is to aggregate predictions in adversarial settings and minimize errors through dynamic mass updates.

**Key Features:**
- **Dynamic Mass Updates:** Experts' masses are adjusted based on their performance, promoting more reliable experts over time.
- **Coarsened Predictions:** Aggregated predictions are derived from the experts’ outputs, ensuring a collective decision-making process.
- **Adversarial Robustness:** Designed to function effectively in adversarial environments, aiming to minimize misclassifications through careful mass allocation.
- **Non-informative Scenarios:** Automatically skips mass updates when all experts agree, avoiding unnecessary adjustments in consensus situations.

---

## 3. Evidence Weighted Majority (EWM)
**Description:**  
The EWM class implements a learning algorithm with multiple `SGDClassifier` experts. Initially, it assigns equal mass to each expert and makes aggregated predictions based on their outputs. If the coarsened prediction results in an abstention or an error, the masses of incorrect experts are reduced by a factor of \( \beta > 0 \), and the residual mass is allocated to the remaining experts to ensure that the total mass sums to 1. No updates are made when all experts agree, as this situation is considered uninformative. The goal is to aggregate predictions in adversarial settings and minimize errors through dynamic mass updates.

**Key Features:**
- **Dynamic Mass Adjustments:** Experts’ masses are updated based on their performance, allowing for more reliable experts to maintain higher masses over time.
- **Coarsened Predictions:** Aggregated predictions are derived from the outputs of the experts, enabling a collective decision-making process.
- **Adversarial Robustness:** Specifically designed to operate effectively in adversarial environments, minimizing errors through strategic mass allocations.
- **Informativeness Check:** Automatically skips mass updates when all experts agree, avoiding unnecessary adjustments in consensus scenarios.
- **Error Reduction Mechanism:** The use of a scaling factor \( \beta \) for updating masses provides a controlled method to decrease the influence of incorrect predictions while redistributing mass among remaining experts.

---

## 4. Evidence Weighted Majority with Normalization (EWMWN)
**Description:**  
The EWMWN class implements a learning algorithm with multiple `SGDClassifier` experts. Initially, it assigns equal mass to each expert. It makes aggregated predictions based on the experts' outputs. If the coarsened prediction results in an error, the masses of incorrect experts are reduced by a factor of \( \beta \), and then all the experts' masses are normalized to maintain the sum of the masses at 1. No updates are made when all experts agree, as the situation is considered uninformative. The goal is to aggregate predictions in adversarial settings and minimize errors through dynamic mass updates.

**Key Features:**
- **Dynamic Mass Adjustments:** Experts’ masses are updated based on their performance, allowing for more reliable experts to maintain higher masses over time.
- **Normalization of Masses:** After adjusting the masses of incorrect experts, normalization ensures that the total mass remains equal to 1, preserving the integrity of the mass function.
- **Coarsened Predictions:** Aggregated predictions are derived from the outputs of the experts, enabling a collective decision-making process.
- **Adversarial Robustness:** Specifically designed to operate effectively in adversarial environments, minimizing errors through strategic mass allocations.
- **Informativeness Check:** Automatically skips mass updates when all experts agree, avoiding unnecessary adjustments in consensus scenarios.

---

## 5. Evidence Consistent (EC)
**Description:**  
The Evidence Consistent Algorithm (EC) is a robust learning framework designed to leverage multiple `SGDClassifier` experts alongside a perfect expert. It focuses on dynamically adjusting the mass allocations among subsets of experts based on their prediction accuracy in adversarial environments. The algorithm aims to minimize prediction errors while effectively aggregating expert outputs.

**Key Features:**
- **Multi-Expert Framework:** Incorporates multiple `SGDClassifier` models along with a perfect expert, enhancing predictive capability in adversarial settings.
- **Dynamic Mass Allocation:** Initializes a mass function with a total mass of 1 for the entire set of experts. Masses are updated according to expert predictions and accuracy using the Dempster-Shafer rule, allowing for adaptive responses to varying prediction scenarios.
- **Coarsening Mechanism:** Employs a coarsening function to aggregate predictions from active experts, returning masses for predictions of 1, 0, or abstention, which helps manage uncertainty.
- **Handling Abstentions:** Updates masses allocated to subsets of experts when the coarsened prediction results in an abstention, ensuring continuous adaptation to expert performance.
- **Conflict Management:** Calculates the degree of conflict (K) when combining current and refined masses, allowing for normalization and improved accuracy in mass updates.
- **Perfect Expert Integration:** The perfect expert provides a baseline for evaluation, contributing to the system's ability to recognize and learn from its mistakes effectively.

---

## 6. Evidence Soft Consistent (ESC)
**Description:**  
The Evidence Soft Consistent Algorithm (ESC) leverages multiple `SGDClassifier` experts to dynamically aggregate predictions in adversarial settings. The algorithm begins with an initial mass of 1 assigned to the entire set of experts. It updates the masses of expert subsets based on their prediction accuracy, utilizing the Shafer-Dempster rule to refine the masses. When the coarsened prediction results in an abstention or error, the masses for the subsets are adjusted accordingly. If all experts agree, no updates are made, as this situation does not provide new information. The overarching goal is to minimize errors through adaptive mass updates.

**Key Features:**
- **Dynamic Mass Allocation:** Maintains and updates a mass function representing the confidence in subsets of experts, allowing for flexible responses to prediction outcomes.
- **Shafer-Dempster Rule Utilization:** Refines masses based on the accuracy of expert predictions, where correct predictions receive a mass of \( \alpha \) and incorrect ones receive \( 1 - \alpha \).
- **Handling Abstentions and Errors:** Specifically addresses the outcomes of abstentions and errors by adjusting the masses allocated to expert subsets, enhancing robustness against adversarial influences.
- **Non-informative Consensus Handling:** Recognizes the lack of diversity when all experts make the same prediction and refrains from updating the mass function, acknowledging the uninformative nature of such consensus.
- **Adversarial Robustness:** Designed to operate effectively in challenging, adversarial environments by generating adversarial labels and aggregating predictions accordingly.

## Installation

This project requires Python version 3.10.12 and the following libraries:

- **NumPy** (version >= 1.22.0)
- **Scikit-learn** (version >= 0.24)
- **itertools** (part of the Python standard library, no installation needed)

