import data
import models
import parameters

if __name__ == '__main__':
    df = data.get_prepared_data()  # get clean dataset
    num_parameters, categories = parameters.get_important_parameters(df)
    model = models.get_model(df, num_parameters)  # get best model
