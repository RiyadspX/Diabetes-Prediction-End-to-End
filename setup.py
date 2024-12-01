from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path:str)->List[str]:
    """ A function to return a list of librairies
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines() # readlines reads line by line and then will consider the \n also.
        requirements = [req.replace("\n", "") for req in requirements]  # removing the \n
        
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements


setup(
    name="Diabete classification",
    version="0.0.1",
    author="Riyad",
    author_email="riyadouldabdallah@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
    # create a function to get the requirements 
)


