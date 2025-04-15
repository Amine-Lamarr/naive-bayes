# dataset link : 
# https://www.kaggle.com/uciml/sms-spam-collection-dataset
from sklearn.preprocessing import LabelEncoder 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score , confusion_matrix , recall_score , classification_report
import pandas as pd

# handling data
data = pd.read_csv(r"C:\Users\lenovo\Downloads\spamham_datasets\spam.csv", encoding='ISO-8859-1')
data = data.rename(columns={'v1':'labels' , 'v2':'text'})
data = data[['labels' , 'text']]
x = data['text']
y = data['labels']
#print(data)

# converting labels (spam/ham) into binary classification
le = LabelEncoder()
y_encoder = le.fit_transform(y)

# converting text to numerical values 
TFI = TfidfVectorizer()
text_encoder = TFI.fit_transform(x)

# splitting data to train and test
x_train , x_test , y_train , y_test = train_test_split(text_encoder , y_encoder , test_size= 0.2 , random_state=23)

# preparing model
model = MultinomialNB()
model.fit(x_train , y_train)

# predictions 
predictions = model.predict(x_test)

# accuracy of results 
train_score = model.score(x_train , y_train)
test_score = model.score(x_test , y_test)
precision = precision_score(predictions , y_test)
recall = recall_score(predictions , y_test)
clr = classification_report(predictions , y_test) 
confusionM = confusion_matrix(predictions , y_test)

# predict a new text 
new_text = ["Congratulations, you've won a free ticket to Disneyland!"]
new_text_enc = TFI.transform(new_text)
text_predict = model.predict(new_text_enc)

# printing results
print(f"train score : {train_score*100:.2f}%" )
print(f"test score : {test_score*100:.2f}%" )
print(f"precision : {precision*100:.2f}%")
print(f"recall score : {recall*100:.2f}%")
print(f"confusion matrix : \n" ,confusionM )
print("summary : \n" , clr)
print("text test :" , new_text)
print(f"Prediction: {le.inverse_transform(text_predict)}")
