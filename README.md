# Hotel No-Show Prediction Project

## Overview

The Hotel No-Show Prediction Project aims to predict whether a guest will show up for their hotel reservation or not, using machine learning algorithms such as Random Forest and XGBoost. The project leverages a variety of tools for data preprocessing, feature engineering, model training, evaluation, and visualization. The goal is to provide actionable insights to hotel managers on how to optimize booking processes and reduce the impact of no-shows.

### Features

* **Data Preprocessing**: Handles missing data, scales numerical features, and one-hot encodes categorical variables.
* **Model Training**: Implements Random Forest and XGBoost classifiers for predictive modeling.
* **Model Evaluation**: Provides precision, recall, F1-score, ROC-AUC scores, and confusion matrices for performance evaluation.
* **Visualization**: Plots feature importance and confusion matrices for easy interpretation of results.

## Getting Started

These instructions will help you set up and run the project on your local machine for development and testing purposes.

### Prerequisites

Before running the project, you need to install the required dependencies. You can use the provided `requirements.txt` file to do so:

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/kengsengwang/Hotel-No-Show-Prediction.git
   ```
2. Navigate to the project directory:

   ```bash
   cd Hotel-No-Show-Prediction
   ```
3. Install the required libraries:

   ```bash
   pip install -r requirements.txt
   ```

### Project Structure

The project is structured as follows:

```
root/
├── run.sh                   # Shell script to run the entire pipeline
├── data/
│   └── cleaned_noshow_data.csv  # Dataset (make sure the file exists in the correct location)
├── src/
│   ├── data_preparation.py    # Data loading and preprocessing
│   ├── model_train.py         # Model training script
│   ├── model_evaluation.py    # Model evaluation and performance metrics
│   ├── data_visualization.py  # Visualization tools for model performance
│   ├── hyperparameter_tuning.py # Hyperparameter tuning for Random Forest and XGBoost
│   ├── input_handler.py       # Script for handling user input (file paths, etc.)
│   ├── main.py                # Main script that ties everything together
│   ├── predict.py             # Script for making predictions using trained models
│   └── random_forest_noshow.py # Random Forest model training with GridSearchCV
├── requirements.txt           # List of dependencies
├── README.md                  # Project documentation
└── LICENSE                    # License file (MIT License)
```

### Running the Project

To run the project and train the models:

1. **Train and evaluate models** using the `run.sh` script:

   ```bash
   ./run.sh
   ```

2. **Or** manually run the steps:

   * **Train models**:

     ```bash
     python src/model_train.py
     ```
   * **Evaluate models**:

     ```bash
     python src/model_evaluation.py
     ```
   * **Visualize results**:

     ```bash
     python src/data_visualization.py
     ```

3. **Make predictions** using a trained model:

   ```bash
   python src/predict.py
   ```

### Hyperparameter Tuning

The `hyperparameter_tuning.py` script allows you to perform grid search for optimizing the model parameters. You can run it manually if you want to tune the models:

```bash
python src/hyperparameter_tuning.py
```

### Input Handling

The `input_handler.py` script manages file path inputs to ensure that the correct files are used when running the scripts. You will be prompted to input the file paths when necessary.

---

## Future Work

If time, cost, and energy permit, the following areas could be improved:

* **Model Optimization**: Experiment with more advanced models like XGBoost or Neural Networks for higher accuracy.
* **Hyperparameter Tuning**: Use techniques like GridSearchCV or RandomizedSearchCV to fine-tune model parameters.
* **Feature Engineering**: Incorporate additional features or external datasets to improve the model's performance.
* **Deployment**: Deploy the model as a real-time API service for hotel no-show predictions.

---

## Contributions

Feel free to fork the repository and submit pull requests with improvements or fixes. All contributions are welcome!

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

For any inquiries or feedback, please reach out to [wangkengseng@gmail.com](mailto:wangkengseng@gmail.com) or visit my [LinkedIn Profile](https://www.linkedin.com/in/wang-keng-seng-b5168221/).

