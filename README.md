# Amadeus

This is the project motivated by Amadeus APP which appears in "Steins-gate".

I implemented chatbot using transformer model and provide the whole code + android application source code.

If you are a programmer who are willing to develop own chatbot, get the code and freely modify it!

# Requirements for development

Knowledge of android programming skill (with Java) and python programming skill (with tensorflow).

Public IP or Private IP (We will use the server, so you need IP and PORT of the server)

Test Environment (If your compiler shows that some functions are not available, check the environment.)
- tensorflow ver. 2.4.1
- tensorflow_datasets ver. 4.0.1
(tfds.features.text.SubwordTextEncoder => tfds.deprecated.text.SubwordTextEncoder in higher tensorflow version like 2.4.1)
- android studio ver. 4.0
- python 3.6.10

If you have any errors, let me know.

# Introduction
I recommend you to watch this video before going further.
https://www.youtube.com/watch?v=HGiEBDDjy-I

The structure of the project is very simple. 

There are one chatbot server and one android client (= android application). 

When you activate the android application on your phone (or android studio) and enter chatting room, android client start to connect with chatbot server.

If you click the send button, the text you typed will go to chatbot server. Then, the AI module in the chatbot server produce a answer text according to your text.

The answer text go to your android application.

Finally, you can see what AI answered on your phone.

# How to start android application?

1. Install the android studio (If you already have one, skip this)
2. Open the folder "./Chatbot_android_application_client/Amadeus" as an android project
3. Find file "./Chatbot_android_application_client\Amadeus\app\src\main\java\com\example\amadeus\NetworkConfigure.java" and insert your own public or private ip to the variable string IP. (ex. IP = "192.168.0.1") The PORT number is decided by Chatbot server, so you can just leave it now. 
4. Click the run button in your android studio 

# How to start chatbot server?

1. Go to the folder "./Chatbot_core"
2. Find file "Chatbot.py".
3. Change the variables named HOST and PORT. You can assign public ip or private ip to the HOST and assign any available port to the PORT. Note that, the port number should be same with the port number in android application (Check How to start android application?) 
2. Type the "python Chatbot.py"

# How to make my own chatbot?

1. Go to the folder "./Chatbot_core/Data"
2. There is a dialogue_example.txt. If you want to teach some other words, insert more texts to that text file.
Warning:
  You should keep the form of the text. There are three categories (name, text, sentiment) and each category is separated by tab (you know, tab on the keyboard).
  There must be one question and following one answer! The application doesn't consider whole context. <br>
4. Type the "python Chatbot.py --training"

# Copyright
There are some copyright issues with images. 
If you want to use this for commercial, you should replace all the images in the project. (I wrote the code, but the images are not mine)

This project is just for reference and education.
