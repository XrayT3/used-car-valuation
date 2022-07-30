# used-car-valuation
Data Analysis project for valuation used car

I tried to find a correlation between the parameters of the car and the price at which it was sold.
In order to create models that can predict the price based on auto parameters.

File `data.py` downloads the .csv file and prepares the data(fills in empty cells, converts to a convenient format)  
File `parameters.py` is necessary in order to filter out unnecessary categories and leave only those that most strongly affect the price.  
The last file `models.py` creates models with maximum accuracy for price prediction.

This project was completed during the [course](https://cognitiveclass.ai/courses/data-analysis-python)
