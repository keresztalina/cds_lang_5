# Assignment 5 - Detecting depression in text
This assignment is ***Part 5*** of the portfolio exam for ***Language Analytics S23***. The exam consists of 5 assignments in total (4 class assignments and 1 self-assigned project). This is the self-assigned project.

## Contribution
This assignment was created making use of code provided as part of the course. The final code is my own. 

Here is the link to the GitHub repository containing the code for this assignment: https://github.com/keresztalina/cds_lang_5

## Assignment description
In this self-assigned project, I wanted to see how well I could apply the methods learned during this course to a new dataset. The new dataset concerns depression, and makes use of short texts from the website Reddit that either indicate that the author is suffering from depression, or that they are not suffering from depression. There are 7732 texts in total, split equally between texts that contain indications of depression and texts that do not. The classification is binary, and I decided to train a neural net classifier. Here is the source of the data: https://www.kaggle.com/datasets/infamouscoder/depression-reddit-cleaned

## Methods
The purpose of this script is to train a neural net classifier on binary text data, then test the classifier's accuracy. 

The data is first loaded into the script. It has already been cleaned by the uploader, so it only needs to be split into training (90%) and test (10%) sets. 

Secondly, the texts are vectorized using a Tf-Idf vectorizer. The vectorizer allows for unigrams and bigrams, and converts every token into lowercase in order not to distinguish between words at the beginning and middle of a sentence, as well as to account for lack of consistent capitalization in internet texts. The most common and most rare words are discarded, retaining a maximum of 500 features. The vectorizer is saved.



## Usage
### Prerequisites
This code was written and executed in the UCloud application's Coder Python interface (version 1.77.3, running Python version 3.9.2). UCloud provides virtual machines with a Linux-based operating system, therefore, the code has been optimized for Linux and may need adjustment for Windows and Mac.

### Installations
1. Clone this repository somewhere on your device. 
2. The data is already contained within the ```/cds_lang_2/in``` folder.
3. Open a terminal and navigate into the ```/cds_lang_5``` folder. Run the following lines in order to install the necessary packages and load the required language model:
        
        pip install --upgrade pip
        python -m spacy download en_core_web_md
        python3 -m pip install -r requirements.txt

### Run the script.
In order to run the script, make sure your current directory is still the ```/cds_lang_2``` folder. 

In order to run the logistic regression classifier, from command line, run:

        python3 src/LR.py
