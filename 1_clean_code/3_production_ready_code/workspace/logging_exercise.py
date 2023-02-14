"""Exercise Logging functionality."""

## STEPS TO COMPLETE ##
# 1. import logging
# 2. set up config file for logging called `results.log`
# 3. add try except with logging for success or error
#    in relation to checking the types of a and b
# 4. check to see that log is created and populated correctly
#    should have error for first function and success for
#    the second
# 5. use pylint and autopep8 to make changes
#    and receive 10/10 pep8 score

# %%
import logging

# Initialize basic logging to logfile only
# ... file will be created if it doesn't exist
logging.basicConfig(
    filename='results.log', level=logging.INFO, filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def sum_vals(int_a, int_b):
    """Sum two values.

    Args
    ----
        int_a: (int)
        int_b: (int)

    Returns
    -------
        int_a + int_b (int)
    """
    # Check that a and b are ints
    if not isinstance(int_a, int) or not isinstance(int_b, int):
        raise TypeError('Inputs must be integers')
    return int_a + int_b


if __name__ == "__main__":
    logging.info('Starting the program')
    try:
        sum_vals('no', 'way')
        logging.info('SUCESS (1)')
    except TypeError:
        logging.error('ERROR (1)')
    try:
        sum_vals(4, 5)
        logging.info('SUCCESS (2)')
    except TypeError:
        logging.error('ERROR (2)')
    logging.info('Ending the program')


# %%
