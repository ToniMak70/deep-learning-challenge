# Neural Network Model Analysis Report

1) __Overview__ of the analysis: 

Alphabet Soup, a charitable foundation, requires a tool to identify promising grant applicants. To address this, we've developed a predictive model that uses their data to determine the likelihood of an applicant's success if funded.

2) __Results__: 
* Data Preprocessing
    * Target: IS_SUCCESSFUL was set as the prediction target.
    * Features: All other columns (excluding IS_SUCCESSFUL and EIN) were used as input features.
    * Non-Predictive Column Removal: EIN was removed from the dataset as it did not provide useful information for the model.
    
* Compiling, Training, and Evaluating the Model
    1. Model Architecture: Neurons, Layers, and Activation Functions
    * Structure:
        * The nn_3layer(optimized) neural network comprises three hidden layers.
        * Layer 1: 7 neurons.
        * Layer 2: 14 neurons.
        * Layer 3: 21 neurons.
        * Output Layer: 1 neuron.
    * Activation Functions:
        * Hidden Layers: relu (Rectified Linear Unit). relu was chosen for its efficiency and effectiveness in mitigating the vanishing gradient problem, common in deep neural networks.
        * Output Layer: sigmoid. The sigmoid function was selected because it outputs a probability between 0 and 1, suitable for binary classification (predicting IS_SUCCESSFUL).
    * Rationale for Layer and Neuron Count:
        * The architecture evolved through experimentation. The initial model, nn_2layer, had two hidden layers. The addition of a third hidden layer in nn_3layer model aimed to capture more complex relationships within the data. 
    2. Target Model Performance Achievement
        * The nn_2laayer model only achieved a predictive accuracy of 73%. The nn_3layer model achieved the target predictive accuracy, exceeding 75%. The final evaluation on the test data yielded an accuracy of approximately 75.68% for the nn_3layer model.
    3. Steps Taken to Increase Model Performance
        * Feature Engineering:
            * Reintroduced the NAME column. Initially, both the EIN and NAME columns were removed, but the NAME column was restored in nn_3layer model to provide potentially valuable additional features.
        * Architectural Adjustments:
            * The nn_2layer model consisted of two hidden layers. The nn_3layer model consisted of three hidden layers, this allowed for the testing of different layer sizes.
        * Learnings from Previous Attempts:
            * The incremental changes to the model architecture and feature selection were informed by the results of the previous models, enabling a data-driven approach to optimization.

3) __Summary__: 

The AlphabetSoupCharity_Optimization.ipynb (or nn_3layer model) achieves a test accuracy of 75.68%, exceeding the 75% target. To potentially improve performance, we recommend a deeper data analysis for enhanced feature selection and engineering. Further experimentation will be crucial for optimizing the model.


![nn_3layer_accuracy_plot](https://github.com/user-attachments/assets/072cd954-c885-4def-adea-f8e8ac9ae569)

