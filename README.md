#Automated-Question-Answering-from-FAQs

This program will help you to create a "Automated Question answering" using Bert sentence embeddings.

First we provide a csv file containing columns of "Questions" and "Answers" from the "Frequently Asked Questions"(FAQs) of any website.
Or you can create your your question and answers.

Now run the program which will download "distilbert-base-nli-stsb-mean-tokens" of Bert.
This takes time when you run this program for the first time to download the package.

Later program asks the user to input a Question, then it identifies the relevant question in the CSV file that matches with the user's question.
And prints out the answer that matched most accurately.

If the Question asked by the user seems unmatching from the questions which we have in the CSV file, then it prints a message to contact through the mail.

To exit, user can enter 'q'.

Note: Adjust the threshold of similarity, so that if question from is not matching from the CSV questions, it prints to write a mail.
