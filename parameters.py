import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt


def get_important_parameters(df):
    # let's try to find correlation between different numeric parameters and price
    # if abs(value) > 0.75 than parameter has a strong enough correlation with the price
    df_corr = df.corr()["price"]
    numeric_parameters = df_corr.index[abs(df_corr) > 0.75].tolist()
    numeric_parameters.remove("price")
    # you can check what it looks like
    sns.regplot(x=numeric_parameters[0], y="price", data=df)
    plt.ylim(0, )  # price cannot be less than zero
    plt.show()

    # let's check, if numeric parameters are statistically significant
    numeric_parameters = [parameter for parameter in numeric_parameters
                          if stats.pearsonr(df[parameter], df["price"])[1] < 0.001]
    # They are all statistically significant, because P-value < 0.001
    print("\nList of numerical parameters that correlate with price and are statistically significant:")
    print(numeric_parameters)

    # now look at the distribution by category and price
    # we have to find parameters with string type
    categories = df.dtypes.index[df.dtypes == "object"].tolist()
    # looking to the plot we should select categories with two or three groups
    # and with big enough difference between them
    sns.boxplot(x=categories[2], y="price", data=df)
    plt.show()
    plt.close()
    # selected categories are below
    categories = ["fuel-type", "aspiration", "drive-wheels"]
    print("\nList of categories:")
    print(categories)

    # let's check, if categories are statistically significant
    grouped = df[["fuel-type", "price"]].groupby(["fuel-type"])
    f_val, p_val = stats.f_oneway(grouped.get_group("diesel")["price"], grouped.get_group("gas")["price"])
    print("ANOVA results for 'fuel-type': F=" + str(f_val) + ", P=" + str(p_val))
    grouped = df[["aspiration", "price"]].groupby(["aspiration"])
    f_val, p_val = stats.f_oneway(grouped.get_group("std")["price"], grouped.get_group("turbo")["price"])
    print("ANOVA results for 'aspiration': F=" + str(f_val) + ", P=" + str(p_val))
    grouped = df[["drive-wheels", "price"]].groupby(["drive-wheels"])
    f_val, p_val = stats.f_oneway(grouped.get_group("rwd")["price"], grouped.get_group("fwd")["price"])
    print("ANOVA results for 'drive-wheels': F=" + str(f_val) + ", P=" + str(p_val))

    # only "drive-wheels" has small enough P-value and large F-value
    print("\nList of important parameters:")
    print(numeric_parameters + ["drive-wheels"])
    return numeric_parameters, ["drive-wheels"]
