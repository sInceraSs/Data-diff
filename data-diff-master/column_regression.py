from sklearn import linear_model
import numpy as np
import pandas as pd


def regression(origin_column, dest_column):
    """
    Perform the regression, using one of the
    custom defined classes (see below).

    Return an instance of the class below.
    """
    # input for x needs to be of shape (n_observations,n_features)
    # for us n_features will be 1
    # if using pandas columns -- see this
    # https://github.com/justmarkham/DAT4/blob/master/notebooks/08_linear_regression.ipynb
    # could also use numpy linalg leastsquares
    o_c = np.reshape(origin_column.values, (len(origin_column), 1))
    cur_mod = LinearModification(o_c, dest_column)
    cur_mod.run()
    return cur_mod


class LinearModification(object):
    """
    This class represents an object that could be used
    to search for modifications

    This class will only search for linear regression,
    but if your dataset would have other modifications,
    build a class with the same interface as this one,
    and you can use it in the regression function
    defined above.
    """

    def __init__(self, origin, dest):
        """
        Initialize the object with the origin and
        destination columns
        """
        self.reg = linear_model.LinearRegression()
        self.origin = origin
        self.dest = dest

    def __str__(self):
        """
        If you choose to use your own class,
        write a string method that would show the
        modifications this class detected on the columns

        Note that calling this before calling run
        is not intended, as the data of the modification
        will not yet have been calculated.
        """
        return 'linear modification with coef {:.2f} and intercept {:.2f}'.\
            format(self.reg.coef_[0], self.reg.intercept_)

    def run(self):
        """
        Run the actual regression model
        """
        self.reg.fit(self.origin, self.dest)

    def quality(self):
        """
        Determine the quality of an assignment
        This function is used to determine whether
        assignment was satisfactory or not
        
        Note that this quality function
        looks for a perfect assignment
        (with a tolerance for floating point)
        Uses numpy's allclose function

        Returns a boolean.
        """
        # apply the fitted regression to o_c
        # see if its dest_column
        # return a quality measure
        pred = self.reg.predict(self.origin)
        qual = np.allclose(pred, self.dest.values)
        return qual


if __name__ == "__main__":
    o, d = (pd.Series([0, 1, 2]), pd.Series([0, 2, 4]))
    x = regression(o, d)
    print(x)
    x.quality()
