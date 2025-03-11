import pickle
import warnings
warnings.filterwarnings('ignore')

def load_model(filename):
    """ Load the trained model from a pickle file. """
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model

def load_label_encoder(encoder_filename):
    """ Load the LabelEncoder from a pickle file. """
    with open(encoder_filename, 'rb') as file:
        label_enc = pickle.load(file)
    return label_enc

def predict_with_model(model, label_enc, user_input):
    """ Make a prediction and return the species name. """
    prediction = model.predict([user_input])
    species_name = label_enc.inverse_transform(prediction)  
    return species_name[0]

def main():
    model_filename = 'trained_model.pkl'  
    model = load_model(model_filename)
    label_enc = load_label_encoder('label_encoder.pkl')

    user_input = [5.8, 3, 4.3, 1.2]
    prediction = predict_with_model(model, label_enc, user_input)
    print(f"The predicted output is: {prediction}")

if __name__ == "__main__":
    main()
