# Artilizer
The Article Visualizer

## Introduction

Q) What is Artilizer ?

 - Got bored of reading boring article? 

A) Watch them instead of reading *via* .**Visualizing Contents**.

## Requirements

 - Python 3
 - Pip
 - NLTK
 - opencv
 - Django
 - moviepy
 - sklearn
 - pandas

## How to use

### Setting Up Development Server (Virtual Environment)

 - Clone the project using `git clone` command.
 - Type `cd Artilizer`
 - If virtualenv is not installed, run sudo apt-get install pipenv followed by:
	`pipenv install | pipenv shell (Ubuntu / Mac)`
 - Run `python manage.py migrate`.
 - Run `python manage.py collectstatic`
 - Start the server using `python manage.py` runserver and visit `http://localhost:8000`

### Basic (EverydayUtilities Dashboard)

 - Start the server using `python manage.py`
 - Navigate to `http://localhost:8000`
 - Enter the Article to transform it to visualizing content.


## Todo

 - Work on the Sentiment Analyser
 - Article Summarizer Efficiency
