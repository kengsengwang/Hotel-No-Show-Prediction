from data_preprocessing import preprocess_data
from model_training import train_model
from model_evaluation import evaluate_model

def main():
    data = preprocess_data()
    model = train_model(data)
    evaluate_model(model)

if __name__ == "__main__":
    main()
