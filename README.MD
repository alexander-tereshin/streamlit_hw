# Bank Client Classification Project
A simple Streamlit service to predict the reaction of bank clients to new features. It based on linear classifier trained on bank data. 
This project serves as a homework assignment for the ML and High Load Systems course at the Higher School of Economics.

Try app [here](https://bank-classification.streamlit.app/)!

Features
---
The user can enter the parameters of a bank client, such as age, gender, education, income, loans, etc. and get a prediction of whether the client will 
respond positively or negatively to a new feature offer. The user can also see some analytics and details about the model and the data.

  
Installation
----
1. Clone the repository:

```bash
$ git clone https://github.com/alexander-tereshin/streamlit_hw.git
```
2. Install the required dependencies:

```bash
$ pip install -r requirements.txt
$ python -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```
3. Run the app
```bash
streamlit run main.py
```


Usage
---
Open http://localhost:8501 to view the app.


License
---
This project is licensed under the terms of the MIT license.
