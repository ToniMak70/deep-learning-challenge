# deep-learning-challenge

### Background
> The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup. From Alphabet Soup’s business team, you have received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as:
* __EIN__ and __NAME__—Identification columns
* __APPLICATION_TYPE__—Alphabet Soup application type
* __AFFILIATION__—Affiliated sector of industry
* __CLASSIFICATION__—Government organization classification
* __USE_CASE__—Use case for funding
* __ORGANIZATION__—Organization type
* __STATUS__—Active status
* __INCOME_AMT__—Income classification
* __SPECIAL_CONSIDERATIONS__—Special considerations for application
* __ASK_AMT__—Funding amount requested
* __IS_SUCCESSFUL__—Was the money used effectively

## Deliverables
This challenge will be executed using Google Colab and completed by performing the following 5 steps per challenge instructions:
* Preprocess the Data
* Compile, Train, and Evaluate the Model
* Optimize the Model
* Write a Report on the Neural Network Model
* Copy Files Into a Repository

### Step-1: Preprocess the Data
In this step I'll use my knowledge of Pandas and scikit-learn’s `StandardScaler()`, to preprocess the dataset, following the instructions outlined below for the initial model called `Alphabet_2layer.ipynb`. These steps prepare the data for Step 2, where I'll compile, train, and evaluate the neural network model.

Start by uploading the starter file to Google Colab, then using the information we provided in the Challenge files, follow the instructions to complete the preprocessing steps.
1) Read in the `charity_data.csv` to a Pandas DataFrame, and be sure to identify the following in your dataset:
  * What variable(s) are the target(s) for your model?
  * What variable(s) are the feature(s) for your model?
2) Drop the `EIN` and `NAME` columns.
3) Determine the number of unique values for each column.
4) For columns that have more than 10 unique values, determine the number of data points for each unique value.
5) Use the number of data points for each unique value to pick a cutoff point to bin "rare" categorical variables together in a new value, `Other`, and then check if the binning was successful.
6) Use `pd.get_dummies()` to encode categorical variables.
7) Split the preprocessed data into a features array, `X`, and a target array, `y`. Use these arrays and the `train_test_split` function to split the data into training and testing datasets.
8) Scale the training and testing features datasets by creating a `StandardScaler` instance, fitting it to the training data, then using the `transform` function.


### Step-2: Compile, Train, and Evaluate the Model
In this step, a neural network (deep learning model) will be designed to predict the success of Alphabet Soup-funded organizations using the provided dataset. This involves determining the input features, network architecture (neurons and layers), and subsequently compiling, training, and evaluating the model's accuracy and loss.
1) Continue using the file in Google Colab in which you performed the preprocessing steps from Step 1.
2) Create a neural network model by assigning the number of input features and nodes for each layer using TensorFlow and Keras.
3) Create the first hidden layer and choose an appropriate activation function.
4) If necessary, add a second hidden layer with an appropriate activation function.
5) Create an output layer with an appropriate activation function.
6) Check the structure of the model.
7) Compile and train the model.
8) Create a callback that saves the model's weights every five epochs.
9) Evaluate the model using the test data to determine the loss and accuracy.
10) Save and export your results to an HDF5 file. Name the file `AlphabetSoupCharity.h5`.


### Step-3: Optimize the Model
Following the initial model run, optimization will be conducted to achieve a target predictive accuracy exceeding 75%. While multiple iterations may be required, the process will be limited to a maximum of three attempts. The following strategies will guide the optimization process.

Use any or all of the following methods to optimize your model:
* Adjust the input data to ensure that no variables or outliers are causing confusion in the model, such as:
    * Dropping more or fewer columns.
    * Creating more bins for rare occurrences in columns.
    * Increasing or decreasing the number of values for each bin.
    * Add more neurons to a hidden layer.
    * Add more hidden layers.
    * Use different activation functions for the hidden layers.
    * Add or reduce the number of epochs to the training regimen.

1) Create a new Google Colab file and name it `AlphabetSoupCharity_Optimization.ipynb`.
2) Import your dependencies and read in the `charity_data.csv` to a Pandas DataFrame.
3) Preprocess the dataset as you did in Step 1. Be sure to adjust for any modifications that came out of optimizing the model.
4) Design a neural network model, and be sure to adjust for modifications that will optimize the model to achieve higher than 75% accuracy.
5) Save and export your results to an HDF5 file. Name the file `AlphabetSoupCharity_Optimization.h5`.


### Step-4: Write a Report on the Neural Network Model
This section of the challenge requires the creation of a performance report for the Alphabet Soup deep learning model. The report, `AlphabetSoup_Analysis.md`, will be written in Markdown and adhere to the challenge instructions, addressing all specified questions.
1) __Overview__ of the analysis: Explain the purpose of this analysis.
2) __Results__: Using bulleted lists and images to support your answers, address the following questions:
* Data Preprocessing
    * What variable(s) are the target(s) for your model?
    * What variable(s) are the features for your model?
    * What variable(s) should be removed from the input data because they are neither targets nor features?
* Compiling, Training, and Evaluating the Model
    * How many neurons, layers, and activation functions did you select for your neural network model, and why?
    * Were you able to achieve the target model performance?
    * What steps did you take in your attempts to increase model performance?
3) __Summary__: Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and then explain your recommendation.


### Step-5: Copy Files Into This Repository
1) Download your Colab notebooks to your computer.
2) Move them into your Deep Learning Challenge directory in your local repository.
3) Push the added files to GitHub.



## File Structure
deep-learning-challenge/
├── h5_Models/
├── ├── AlphabetSoupCharity (1).h5, AlphabetSoupCharity_Optimization.h5/   
├── Images/
├── ├── nn_2layer_accuracy_plot.png, nn_2layer_loss_plot.png, nn_3layer_accuracy_plot.png, nn_3layer_loss_plot.png/
├── Alphabet_2layer.ipynb
├── AlphabetSoupCharity_Optimization.ipynb
├── LICENSE
└── README.md


## Acknowledgements
- Classmates: I recieved coding and debuging help from multiple classmates during breakout sessions.
- Internet Search: I utilized Google Search and slack overflow to research coding concepts, algorithms, and best practices.
- Support Staff: I recieved help from my instructor and class TA during office hours for help with debugging errors and further explanation for complex code segments.
- AI support: I leveraged Gemini AI and ChatGPT to generate code suggestions, debug errors, and provide explanations for complex code segments.
- Please note: While these tools were invaluable in my development process, the final code is the result of my analysis, testing, and refinement.
