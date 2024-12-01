import sys  # track error details
from src.logger import logging


def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()  # On which file line number.. the exception has occured
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)

    )
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message

# Checking with an example : 
"""
if __name__ == "__main__":
    
    def divide_numbers(a, b):
        try:
            result = a / b
        except Exception as e:
            # Raise a CustomException instead of the original exception
            raise CustomException(e, sys) 

    divide_numbers(10, 0)
  
    try:
        a = 13 / 0
        logging.info('division by zero')
    except Exception as e:
        
        raise CustomException(e, sys)

        """
