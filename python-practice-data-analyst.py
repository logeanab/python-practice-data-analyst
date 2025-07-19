#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 12 16:23:32 2025

@author: acelogeanbueno
"""

#Practicing Python coding

#Section 1: Variables & Data Types
#Print your name and age
name = "Ace"
age = 31
print("My name is", name)
print("I am", age, "years old")

#Swap two numbes
a = 5
b = 10
a, b = b, a
print("a =", a)
print("b =", b)

#Check data types of multiple variables
x = 10
y = 3.14
z = "Hello"
flag = True
print(type(x))
print(type(y))
print(type(z))
print(type(flag))

#Convert float to int and string
num = 9.99
print(int(num))
print(str(num))

#Calculate the area of a rectangle
length = 10
width = 5
area = length * width
print("Area:", area)

#Section 2: Control Flow - If, For, While

#Check if a number is positive, negative, or zero
num = int(input("Enter a number: "))
if num > 0:
    print("Positive")
elif num < 0:
    print("Negative")
else:
    print("Zero")
    
#Even or Odd
n = int(input("Enter a number:"))
if n % 2 == 0:
    print("Even")
else:
    print("Odd")
    
#Print numbers from 1 to 10 using for loop
for i in range(1, 11):
    print(i)
    
#Calculate the sum of numbers from 1 to 100
total = 0
for i in range(1, 10):
    total += i
print("Sum:", total)

#Multiplication table of a given number
n = int(input("Enter a number: "))
for i in range(1, 11):
    print(f"{n} x {i} = {n*i}")
    
#Factorial using while loop
num = int(input("Enter a number: "))
fact = 1
while num > 0:
    fact *= num
    num -= 1
print("Factorial:", fact)

#Check if number divisible by both 3 and 5
n = int(input("Enter a number: "))
if n % 3 == 0 and n % 5 == 0:
    print("Divisible by both 3 and 5")
else:
    print("Not divisible by both")
    
#Section 3: Fuctions and Modules

#Fucntion to add two numbers
def add(a, b):
    return a + b
print(add(10, 5))

#Function to find max of 3 numbers
def maximum(a, b, c):
    return max(a, b , c)
print(maximum(4, 10, 7))

#Function to check if a number is prime
def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n**0.5)+1):
        if n % i == 0:
            return False
        return True
print(is_prime(22))

#Creat a custom greeting using a function
def greet(name):
    print("Hello", name, "Welcome to Python!")

greet("Ace")

#Section 4: Lists & Dictionaries

#Store 5 fruits in a list and print them
fruits = ["apple", "banana", "cherry", "orange", "grape"]
for fruit in fruits:
    print(fruits)

#Add and remove items in a list
colors = ["red", "blue", "green"]
colors.append("yellow")
colors.remove("blue")
print(colors)

#Creat a dictionary of student marks
marks = {"Math": 85, "Science": 90, "English": 80}
print("Science Marks:", marks["Science"])

#Loop through dictionary items
person = {"name": "Ace", "age": 31, "city": "Manila"}
for key, value in person.items():
    print(key, ":", value)
    
#Section 5: Tuples & Sets
    
#Create and print a tuple
colors = ("red", "green", "blue")
print(colors)

#Access elements in a tuple
numbers = (10, 20, 30)
print(numbers[1])

#Tuple unpacking
person = ("Ace", 31, "Kuala Lumpur")
name, age, city = person
print(name, age, city)

#Check if item exists in a tuple
fruits = ("apple", "banana", "orange")
print("banana" in fruits)

#Create a set and print it
languages = {"Python", "Java", "C++"}
print(languages)

#Add item to a set
skills = {"Excel", "SQL"}
skills.add("Python")
print(skills)

#Remove item from a set
skills = {"Python", "Excel", "SQL"}
skills.remove("Excel")
print(skills)

#Set operations: union & intersection
a = {1, 2, 3}
b = {3, 4, 5}
print("Union:", a | b)
print("Intersection:", a & b)

#Find unique elements from a list using a set
nums = [1, 2, 2, 3, 3, 4, 4]
unique = set(nums)
print(unique)

#Check if two sets are disjoint
a = {1, 2, 3}
b = {4, 5}
print(a.isdisjoint(b)) #True

#Section 6: More Functions & Loops

#Create a function to find square of a number
def square(n):
    return n * n
print(square(6))

#Create a function to check if a string is palindrome
def is_palindrome(s):
    return s == s[::-1]
print(is_palindrome("madam"))

#Sum all elements in a list
nums = [1, 2, 3, 4, 5]
print(sum(nums))

#Find the largest number in a list
numbers = [10, 25, 17, 9]
print(max(numbers))

#Count vowels in a string
s = "Hello World"
vowels = "aeiouAEIOU"
count = sum(1 for char in s if char in vowels)
print("Vowel count:", count)

#Find the factorial using recursion
def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)
print(factorial(5))

#Create a simple calculator
def calculator(a, b, operator):
    if operator == "+":
        return a + b
    elif operator == "-":
        return a - b
    elif operator == "*":
        return a * b
    elif operator == "/":
        return a / b
    else:
        return "Invalid operator"
print(calculator(10, 5, "/"))

#Loop through a string and print characters
word = "Python"
for char in word:
    print(char)
    
#Display numbers from 1 to 50 that are divisible by 7
for i in range(1, 51):
    if i % 7 == 0:
        print(i)
    
#Section 7: Lists - Advanced Practice

#Reverse a list manually
nums = [1, 2, 3, 4, 5]
reversed_nums = []
for i in nums[::-1]:
    reversed_nums.append(i)
print(reversed_nums)

#Sort a list in descending order
data = [10, 5, 8, 2]
data.sort(reverse=True)
print(data)

#Remove duplicates from a list
items = [1, 2, 2, 3, 3, 3, 4]
unique_items = list(set(items))
print(unique_items)

#Merge two lists
a = [1, 2, 3]
b = [4, 5]
merged = a + b
print(merged)

#Find common elements in two lists
a = [1, 2, 3, 4]
b = [3, 4, 5, 6]
common = list(set(a) & set(b))
print(common)

#Find the second largest number in a list
nums = [10, 20, 4, 45, 99]
nums.sort()
print("Second largest:", nums[-2])

#List comprehension: squares of even numbers
squares = [x**2 for x in range(1,11) if x % 2 == 0]
print(squares)

#Find the longest word in a list
words = ["python", "data", "analytics", "AI"]
longest = max(words, key=len)
print("Longest word:", longest)

#Count frequency of each element in a list
items = ["apple", "banana", "apple", "orange", "banana"]
freq = {}
for item in items:
    freq[item] = freq.get(item, 0) + 1
print(freq)

#Remove all even numbers from a list
nums = [1, 2, 3, 4, 5, 6]
filtered = [x for x in nums if x % 2 !=0]
print(filtered)

#Section 8: Working with strings

#Count words in sentence
sentence = "Python is fun and powerful"
words = sentence.split()
print("Word count:", len(words))

#Capitalize the first letter of each word
s = "hello world from python"
print(s.title())

#Replace a word in a sentence
sentence = "I love Python"
new_sentence = sentence.replace("love", "like")
print(new_sentence)

#Check if a string contains on digits
text = "12345"
print(text.isdigit())

#Reverse a string
s = "Python"
print(s[::-1])

#Count how many times a letter appears
word = "banana"
print("a appears", word.count("a"), "times")

#Check if a string is a palindrome(case-insensitive)
s = "Madam"
s = s.lower()
print(s == s[::-1])

#Joint a list of strings into one string
words = ["Python", "is", "awesome"]
result = " ".join(words)
print(result)

#Remove punctuation from a sentence
import string

sentence = "Hello, world! Welcome to Python."
cleaned = ''.join(c for c in sentence if c not in string.punctuation)
print(cleaned)

#Get ASCII value of a character
char = 'A'
print("ASCII of", char, "is", ord(char))

#add a new item to a dictionary
person = {"name": "Ace", "age": 31}
person["city"] = "Manila"
print(person)

#update a dictionary value
person = {"name": "Ace", "age": 31}
person ["age"] = 32
print(person)

#Loop through keys and values
car = {"brand": "Toyota", "model": "Vios", "year": 2020}
for key, value in car.items():
    print(key, "->", value)

#Check if a key exists
student = {"name": "Ella", "grade": 90}
print("grade" in student)

#Remove a key from a dictionary
info = {"name": "John", "email": "john@example.com"}
info.pop("email")
print(info)

#Merge two dictionaries
a = {"x": 1, "y": 2}
b = {"z": 3}
a.update(b)
print(a)

#Count letter frequency in a word using a dictionary
word = "apple"
freq = {}
for letter in word:
    freq[letter] = freq.get(letter, 0) + 1
print(freq)

#Create a dictionary from two lists
keys = ["name", "age"]
values = ["Ace", 31]
data = dict(zip(keys, values))
print(data)

#Sort dictionary by values
scores = {"Anna": 85, "John": 95, "Paul": 78}
sorted_scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
print(sorted_scores)

#Get dictionary keys as list
person = {"name": "Ace", "age": 31}
print(list(person.keys()))

#Section 10: Date and Time

#Get current date and time
from datetime import datetime
now = datetime.now()
print(now)

#Format date and time
from datetime import datetime
now = datetime.now()
print(now.strftime("%Y-%m-%d %H:%M:%S"))

#Extract year, month, day
from datetime import datetime
now = datetime.now()
print("Year:", now.year)
print("Month:", now.month)
print("Day:", now.day)

#Add 7 days to current date
from datetime import datetime, timedelta
future = datetime.now() + timedelta(days=7)
print(future)

#Substract days from a date
from datetime import datetime, timedelta
before = datetime.now() - timedelta(days=30)
print(before)

#Get the day of the week
from datetime import datetime
today = datetime.now()
print(today.strftime("%A"))

#Create a specific date
from datetime import date
d = date(2025, 1, 1)
print(d)

#Get number of days between two dates
from datetime import date
d1 = date(2025, 1, 1)
d2 = date(2025, 6, 28)
diff = d2 - d1
print(diff.days)

#Count how many days until your birthday
from datetime import date
today = date.today()
bday = date(today.year, 10, 17)
if bday < today:
    bday = date(today.year + 1, 10, 17)
print("Days until birthday:", (bday - today).days)

#Convert string to datetime
from datetime import datetime
date_str = "2025-06-28 14:00"
dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M")
print(dt)

#Section 11: Mini Projects

#Tip calculator
bill = float(input("Total bill: "))
tip = float(input("Tip %: "))
total = bill + (bill * tip / 100)
print("Total with tip:", total)

#BMI calculator
weight =  float(input("Weight (kg): "))
height = float(input("Height (cm): "))
bmi = weight / (height ** 2)
print("Your BMI is:", round(bmi, 2))

#Countdown timer
import time
seconds = int(input("Enter time in seconds: "))
while seconds > 0:
    print(seconds)
    time.sleep(1)
    seconds -= 1
print("Time's up!")

#Number guessing game
import random
num = random.randint(1, 10)
guess = int(input("Guess a number (1-10): "))
if guess == num:
    print("Correct!")
else:
    print("Wrong! It was", num)
    
#Rock Paper Scissors
import random
choices = ["rock", "paper", "scissors"]
player = input("Rock, Paper or Scissors? "). lower()
computer = random.choice(choices)
print("Computer chose:", computer)

#Create a simple password generator
import random
import string
length = 8
password = ''.join(random.choices(string.ascii_letters + string.digits, k=length))
print("Password:", password)

#Temperature converter (C to F)
c = float(input("Enter temp in Celsius: "))
f = (c * 9/5) + 32
print("Fahrenheit:", f)

#Simple interest calculator
p = float(input("Principal: "))
r = float(input("Rate: "))
t = float(input("Time: "))
si = (p * r * t) / 100
print("Simple Interest:", si)

#Print the Fibonacci series up to n terms
n = int(input("How many terms? "))
a, b = 0, 1
for _ in range(n):
    print(a)
    a, b = b, a + b

#Check if year is a leap year
year = int(input("Enter a year: "))
if (year % 4 == 0 and year % 100 !=0) or (year % 400 == 0):
    print("Leap year")
else:
    print("Not a leap year")

#Section 12: Logic and Practice

#Find the largest of 3 numbers
a = 10
b = 25
c = 15
print(max(a, b, c))

#Check if a number is Armstrong
num = int(input("Enter number: "))
total = sum(int(digit) ** len(str(num)) for digit in str(num))
print("Armstrong" if total == num else "Not Armstrong")

#Check if a number is prime(again)
n = int(input("Enter number: "))
if n > 1:
    for i in range(2, int(n**0.5)+1):
        if n % i == 0:
            print("Not prinme")
            break
        else:
            print("Prime")
else:
    print("Not prime")
    
#Calculate compound interest
p = 1000
r = 5
t = 2
ci = p * ((1 + r/100)**t - 1)
print("Compound Interest", round(ci, 2))

#Convert seconds into hours, mins, secs
sec = int(input("Enter seconds: "))
hrs = sec // 3600
mins = (sec % 3600) // 60
secs = sec % 60
print(f"{hrs}h:{mins}m:{secs}s")

#Create a multiplication table(1 to 10)
for i in range(1, 11):
    for j in range(1, 11):
        print(i*j, end="\t")
    print()
    
#Find the GCD of two numbers
import math
a = int(input("Enter first number: "))
b = int(input("Enter second number: "))
print("GCD:", math.gcd(a, b))

#Find whether a number is even or odd(using fuction)
def is_even(n):
    return n % 2 == 0
print(is_even(8))

#Count uppercase and lowercase letters in a string
s = "Hello World"
upper = sum(1 for c in s if c.isupper())
lower = sum(1 for c in s if c.islower())
print("Uppercase:", upper)
print("Lowercase:", lower)

#Find sum of digits of a number
num = 12345
digit_sum = sum(int(digit) for digit in str(num))
print("Sum of digits:", digit_sum)

#Another set of codes for practice

#Section 1: Pandas Basics
#Import pandas and read a CSV
import os
print(os.getcwd())


import pandas as pd
df = pd.read_csv('your_file.csv')
print(df.head())

#Check shape and data types
print(df.shape)
print(df.types)

#Get summary statistics
print(df.describe())

#Rename a column
df.rename(columns={'old_name'': 'new_name'}, inplace=True)

#Drop a column
df.drop('column_name', axis=1, inplace=True)

#Filter rows by condition
filtered = df[df['Age'] > 30]

#Check for missing values
print(df.isnull().sum())

#Fill missing values
df['column'] = df['column'].fillna(df['column'].mean())

#Create a new column
df['total'] = df['price'] * df['quantity']

#Sort values
df.sort_values(by='total', ascending=False, inplace=True)

#Section 2: Grouping & Aggregation
#Group by a column and calculate mean
print(df.groupby('region')['sales'].mean())

#Count values in each category
print(df['product'].value_counts())

#Multiple aggregations
df.groupby('category').agg({'sales': 'sum', 'profit': 'mean'})

#Pivot table
pd.pivot_table(df, index='region', columns='category', values='sales', aggfunc='sum')

#Crosstab for frequencies
pd.crosstab(df['gender'], df['purchased'])

#Apply a custom function
df['double'] = df['sales'].apply(lambda x: x * 2)

#Remove duplicates
df.drop_duplicates(inplace=True)

#Reset Index
df.reset_index(drop=True, inplace=True)

#Change data type
df['date'] = pd.to_datetime(df['date'])

#Extract year from datetime
df['year'] = df['date'].dt.year

#Section 3: Numpy
#Create an array
import numpy as np
a = np.array([1, 2, 3])

#Create a 2D array
b = np.array([[1, 2], [3, 4]])

#Generate random numbers
np.random.seed(0)
rand_nums = np.random.rand(5)

#Reshape array
b = np.arange(12).reshape(3, 4)

#Basic statistics
print(np.mean(a), np.median(a), np.std(a))

#Element-wise operations
print(a + 10)

#Boolean filtering
print(a[a > 1])

#Sum across axis
print(b.sum(axis=0))

#Matrix multiplication
c = np.dot(b, b.T)

#Create zeros and ones
np.zeros((2, 3)), np.ones((3, 2))

#Section 4: Data Visualization
#Line plot
import matplotlib.pyplot as plt
plt.plot([1, 2, 3], [2, 4, 6])
plt.title("Line Plot")
plt.show()

#Bar plot
df.groupby('category')['sales'].sum().plot(kind='bar')
plt.show()

#Histogram
df['sales'].plot(kind='hist', bins=10)
plt.show()

#Pie chart
df['region'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.axis('equal')
plt.show()

#Boxplot
import seaborn as sns
sns.boxplot(x='region', y='sales', data=df)

#Scatter plot
plt.scatter(df['sales'], df['profit'])
plt.xlabel('Sales')
plt.ylabel('Profit')
plt.show()

#Heatmap of correlations
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

#Countplot for categorical data
sns.countplot(x='category', data=df)

#Subplots
fig, ax = plt.subplots(1, 2)
df['sales'].plot(kind='hist', ax=ax[0])
df['profit'].plot(kind='box', ax=ax[1])

#Save a plot to file
plt.plot([1, 2, 3])
plt.savefig("plot.png")

#Load sample dataset
from sklearn.datasets import load_iris
iris = load_iris(as_frame=True)
df = iris('data')
df['target'] = iris['target']

#Check correlation
print(df.corr())

#Encode categorical variable
df['encoded'] = df['target'].astype('category').cat.codes

#Basic value imputation
df['sepal length (cm)'].fillna(df['sepal length (cm)'].mean(), inplace=True)

#Train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop('target', axis=1), df['target'], test_size=0.2)

#Build a logistic regression model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)

#Model accuracy
print("Accuracy:", model.score(X_test, y_test))

#Confusion matrix
from sklearn.metrics import confusion_matrix
preds = model.predict(X_test)
print(confusion_matrix(y_test, preds))

#Classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, preds))

#Export results to CSV
results = pd.DataFrame({'Actual': y_test, 'Predicted': preds})
results.to_csv("model_results.csv", index=False)

#Section 1: More Pandas & Data Cleaning
#Capitalize all values in a column
df['city'] = df['city'].str.title()

#Strip whitespace in string columns
df['name'] = df['name'].str.strip()

#Replace specific values
df['status'] = df['status'].replace({'old': 'inactive', 'new': 'active"'})

#Convert column to numeric (with error handling)
df['sales'] = pd.to_numeric(df['sales'], errors='coerce')

#Replace outliers using IQR
Q1 = df['amount'].quantile(0.25)
Q3 = df['amount'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['amount'] >= Q1 - 1.5 * IQR) & (df['amount'] <= Q3 + 1.5 * IQR)]

#Fill missing values with mode
df['gender'] = df['gender'].fillna(df['gender'].mode()[0])

#Extract domain from email
df['domain'] = df['email'].str.split('@').str[1]

#Filter rows with multiple conditions
df[df['age'] > 30 & (df['income'] > 50000)]

#Replace empty strings with NaN
df.replace('', pd.NA, inplace=True)

#Count null values in each row
df['missing'] = df.isnull().sum(axis=1)

#Section 2: DateTime Handling
#Convert string to datetime
df['date'] = pd.to_datetime(df['date'])

#Extract month and day from date
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day

#Calculate days between two dates
df['days_between'] = (df['end_date'] - df['start_date']).dt.days

#Filter data for a specific year
df[df['date'].dt.year == 2024]

#Create datetime from separate columns
df['full_date'] = pd.to_datetime(df[['year', 'month', 'day']])

#Set datetime column as index
df.set_index('date', inplace=True)

#Resample time series data(monthly)
monthly = df['sales'].resample('M').sum()

#Time difference between rows
df['time_diff'] = df['timestamp'].diff()

#Add 30 days to a date column
df['next_month'] = df['date'] + pd.Timedelta(days=30)

#Extract week of year
df['week'] = df['date'].dt.isocalendar().week

#Section 3: Feature Engineering & Encoding
#One-hot encoding
df_encoded = pd.get_dummies(df, columns=['gender', 'city'])

#Binning ages into categories
bins = [0, 18, 35, 60, 100]
labels = ['Teen', 'Young Adult', 'Adult', 'Senior']
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels)

#Label encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['label'] = le.fit_transform(df['category'])

#Count encoding
freq = df['product'].value_counts() / len(df)
df['product_freq'] = df['product'].map(freq)

#Normalize a column
df['normalized_sales'] = (df['sales'] - df['sales'] - df['sales'].mean()) / df['sales'].std()

#Create boolean flags
df['high_spender'] = df['total'] > 1000

#Rank values in a column
df['rank'] = df['score'].rank(ascending=False)

#Creat interaction feature
df['income_per_age'] = df['income'] / df['age']

#Log transformation
import numpy as np
df['log_income'] = np.log1p(df['income'])

#Section 4: Machine Learning Preprocessing
#Scale features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[['x1', 'x2']] = scaler.fit_transform(df[['x1', 'x2']])

#Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Logistic regression
from sklearn.linear_model import LogisticsRegression
model = LogisticRegression()
model.fit(X_train, y_train)

#Predict and get accuracy
from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

#Confusion matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))

#Classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

#KNN classification
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

#Cross-validation score
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, x, y, cv=5)
print("CV Avg Accuracy:", scores.mean())

#Simple Decision Tree
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)

#ROC Curve
from sklearn.metrics import roc_curve, roc_auc_score
probs = model.predict_proba(X_test)[;, 1]
fpr, tpr, _ = roc_curve(y_test, probs)

#Section 5: Real-World Scenarios & Utilities
#Save DataFrame to CSV
df.to_csv("cleaned_data.csv", index=False)

#Load Excel file
df = pd.read_excel("data.xlsx")

#Combine multiple CSVs
import glob
files = glob.glob("data_folder/*.csv")
df = pd.concat([pd.read_csv(f) for f in files])

#Rename multiple columns
df.rename(columns=lambda x: x.lower().replace(" ", "_"), inplace=True)

#Merge two datasets
df_all = pd.merge(df1, df2, on="id", how="left")

#Apply function row-wise
df['full_name'] = df.apply(lambda row: row['first'] + " " + row['last'], axis=1)

#Filter numeric columns only
num_cols = df.select_dtypes(include='number').columns

#Save plot to file
import matplotlib.pyplot as plt
plt.plot([1, 2, 3])
plt.savefig("output.png")

#Profile your data (exploratory)
!pip install pandas-profiling
from pandas_profiling import ProfileReport
profile = ProfileReport(df)
profile.to_file("report.html")

#Zip column values into a list of tuples
tuples = list(zip(df['x'], df['y']))

#New set of python codes
#Section 1: Data Loading & Inspection
#Load a CSV file
import pandas as pd
df = pd.read_csv('your_data.csv')
print(df.head())

#Check data types
print(df.types)

#Get basic info
print(df.info())

#Decribe numeric columns
print(df.describe())

#Count missing values
print(df.isnull().sum())

#Drop rows with missing data
df_clean = df.dropna()

#Fill missing values with mean
df['column_name'] = df['column_name'].fillna(df['column_name'].mean())

#Rename columns
df.rename(columns={'old_name': 'new_name'}, inplace=True)

#Checkunique values
print(df['column_name'].unique())

#Count frequency of each value
print(df['column_name'].value_counts())

#Scetion 2: Data Transformation
#create a new column based on condition
df['high_sales'] = df['Sales'].apply(lambda x: 'High' if x > 300 else 'Low')

#Apply a function to a column
df['Discounted'] = df['Price'].apply(lambda x: x * 0.9)

#Convert data type
df['Date'] = pd.to_datetime(df['Date'])

#Extract year from date
df['Year'] = df['Date'].dt.year

#Filter rows based on condition
high_sales = df[df['Sales'] > 300]

#Sort by column
df_sorted = df.sort_values(by='Sales', ascending=False)

#Group by category and get mean
print(df.groupby('Region')['Sales'].mean())

#Pivot Table
print(df.pivot_table(index='Region', columns='Products', values='Sales', aggfunc='sum'))

#Drop duplicates
df.drop_duplicates(inplace=True)

#Reset index
df.reset_index(drop=True, inplace=True)

#Section 3: Visualization
import matplotlib.pyplot as plt
import seaborn as sns

#Line plot
plt.plot(df['Sales'])
plt.title('Sales Over Time')
plt.show()

#Histogram
df['Sales'].hist()
plt.show()

#Box plot
sns.boxplot(x='Region', y='Sales', data=df)
plt.show()

#Bar chart
df['Region'].value_counts().plot(kind='bar')
plt.show()

#Scatter plot
plt.scatter(df['Price'], df['Sales'])
plt.xlabel('Price')
plt.ylabel('Sales')
plt.show()

#Correlation matrix
sns.heatmap(df.corr(), annot=True)
plt.show()

#Pie chart
df['Region'].value_counts().plot.pie(autopct='%1.1f%')
plt.show()

#KDE plot
sns.kdeplot(df['Sales'], shade=True)
plt.show()

#Count plot
sns.countplot(x='Product', data=df)
plt.show()

#Pairplot
sns.pairplot(df[['Sales', 'Price', 'Profit']])
plt.show()

#Section 4: Feature Engineering & Encoding
#Label encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Product_Code'] = le.fit_transform(df['Product'])

#One-hot encoding
df = pd.get_dummies(df, columns=['Regions'])

#Create interaction feature
df['Price_Sales'] = df['Price'] * df['Sales']

#Bin a numerical column
df['Sales_Bin'] = pd.cut(df['Sales'], bins=3, labels=["Low", "Medium", "High"])

#Normalize a column
df['Price_norm'] = (df['Price'] - df['Price'].mean()) / df['Price'].std()

#Min-max scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df[['Sales_scaled']] = scaler.fit_transform(df[['Sales']])

#Log transform
df['Log_Sales'] = np.log1p(df['Sales'])

#Polynomial feature
df['Sales_squared'] = df['Sales'] ** 2

#Date difference
df['Days_Since_Sale'] = (pd.to_datetime('today') - df['Date']).dt.days

#Custom mapping
map_dict = {'Yes': 1, 'No': 0}
df['Returned'] = df['Returned'].map(map_dict)

#Section 5: Model Preparation
#Select features and target
X = df[['Price', 'Discount', 'Profit']]
y = d['Sales']

#Train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Linear Regression
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

#Predict and evaluate
y_pred = model.predict(X_test)

#Evaluation metrics
from skelearn.metrics import mean_squared_error, r2_score
print("RMSE", mean_squared_error(y_test, y_pred, squared=False))
print("R² Score:", r2_score(y_test, y_pred))

#Cross-validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5)
print("CV Scores:", scores)

#Save model
import joblib
joblib.dump(model, 'linear_model.pkl')

#Load model
model_loaded = joblib.load('linear_model.pkl')

#Predict with new data
print(model_loaded.predict([[200, 20, 50]]))

#Visualize prediction vs actual
plt.scatter(y_test, y_pred)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')
plt.show()






   




    































