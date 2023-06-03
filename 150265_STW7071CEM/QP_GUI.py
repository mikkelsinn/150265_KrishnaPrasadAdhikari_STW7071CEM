from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from tkinter import *
import tkinter.ttk as ttk
from tkinter import scrolledtext
from tkinter import messagebox
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import ujson
import time

p_stemmer = PorterStemmer()
stop_words = stopwords.words('english')
tfidf = TfidfVectorizer()

with open('pub_list_stemmed.json', 'r') as a:
    pub_list_stemmed = ujson.load(a)
with open('pub_dict_indexed.json', 'r') as a:
    pub_index = ujson.load(a)
with open('auth_list_stemmed.json', 'r') as a:
    auth_list_stemmed = ujson.load(a)
with open('auth_dict_indexed.json', 'r') as a:
    auth_index = ujson.load(a)
with open('pub_name.json', 'r') as a:
    pub_name = ujson.load(a)
with open('pub_url.json', 'r') as a:
    pub_url = ujson.load(a)
with open('author.json', 'r') as a:
    pub_cu_author = ujson.load(a)
with open('pub_date.json', 'r') as a:
    pub_date = ujson.load(a)


def search_and_publish(inp=None):
    start_time = time.time()
    outputField.delete(1.0, END)
    outputCount.delete(0, END)
    queryTimeEntry.delete(0, END)
    input_text = inputField.get().strip()

    if len(input_text) == 0:
        messagebox.showinfo(title="Invalid Input!!!", message="Please enter at some words.")
    else:
        search_results = {}
        input_data = input_text.lower().split()

        if len(input_data) < 2:
            messagebox.showinfo(title="Invalid Input!!!", message="Please enter at least 2 words.")
        else:
            valid_input = True
            for token in input_data:
                if len(token) <= 2:
                    valid_input = False
                    break

            if valid_input:
                if rule.get() == 2:
                    outputField.configure(fg='#cc6600')
                    pointer = []
                    for token in input_data:
                        stem_temp = ""
                        stem_word_file = []
                        temp_file = []
                        word_list = word_tokenize(token)

                        for x in word_list:
                            if x not in stop_words:
                                stem_temp += p_stemmer.stem(x) + " "
                        stem_word_file.append(stem_temp)

                        indexed_from = 1
                        if pub_index.get(stem_word_file[0].strip()):
                            pointer = pub_index.get(stem_word_file[0].strip())

                        if len(pointer) == 0:
                            if auth_index.get(stem_word_file[0].strip()):
                                pointer = auth_index.get(stem_word_file[0].strip())
                            indexed_from = 2

                        if len(pointer) == 0:
                            search_results = {}
                        else:
                            if indexed_from == 1:
                                for j in pointer:
                                    temp_file.append(pub_list_stemmed[j])
                            if indexed_from == 2:
                                for j in pointer:
                                    temp_file.append(auth_list_stemmed[j])
                            temp_file = tfidf.fit_transform(temp_file)
                            cosine_output = cosine_similarity(temp_file, tfidf.transform(stem_word_file))

                            if indexed_from == 1:
                                if pub_index.get(stem_word_file[0].strip()):
                                    for j in pointer:
                                        search_results[j] = cosine_output[pointer.index(j)]

                            if indexed_from == 2:
                                if auth_index.get(stem_word_file[0].strip()):
                                    for j in pointer:
                                        search_results[j] = cosine_output[pointer.index(j)]

                else:
                    outputField.configure(fg='#9966ff')
                    indexed_from = 1
                    pointer = []
                    match_word = []
                    for token in input_data:
                        temp_file = []
                        set2 = set()
                        stem_word_file = []
                        word_list = word_tokenize(token)
                        stem_temp = ""
                        for x in word_list:
                            if x not in stop_words:
                                stem_temp += p_stemmer.stem(x) + " "
                        stem_word_file.append(stem_temp)

                        if pub_index.get(stem_word_file[0].strip()):
                            set1 = set(pub_index.get(stem_word_file[0].strip()))
                            pointer.extend(list(set1))

                            if not match_word:
                                match_word = list({z for z in pointer if z in set2 or (set2.add(z) or False)})
                            else:
                                match_word.extend(list(set1))
                                match_word = list({z for z in match_word if z in set2 or (set2.add(z) or False)})
                            indexed_from = 1
                        else:
                            if auth_index.get(stem_word_file[0].strip()):
                                set1 = set(auth_index.get(stem_word_file[0].strip()))
                                pointer.extend(list(set1))

                                if not match_word:
                                    match_word = list({z for z in pointer if z in set2 or (set2.add(z) or False)})
                                else:
                                    match_word.extend(list(set1))
                                    match_word = list({z for z in match_word if z in set2 or (set2.add(z) or False)})
                            else:
                                pointer = []
                            indexed_from = 2

                    if len(input_data) > 1:
                        match_word = {z for z in match_word if z in set2 or (set2.add(z) or False)}

                        if len(match_word) == 0:
                            search_results = {}
                        else:
                            if indexed_from == 1:
                                for j in list(match_word):
                                    temp_file.append(pub_list_stemmed[j])
                            if indexed_from == 2:
                                for j in list(match_word):
                                    temp_file.append(auth_list_stemmed[j])

                            temp_file = tfidf.fit_transform(temp_file)
                            cosine_output = cosine_similarity(temp_file, tfidf.transform(stem_word_file))

                            for j in list(match_word):
                                search_results[j] = cosine_output[list(match_word).index(j)]
                    else:
                        if len(pointer) == 0:
                            search_results = {}
                        else:
                            if indexed_from == 1:
                                for j in pointer:
                                    temp_file.append(pub_list_stemmed[j])
                            if indexed_from == 2:
                                for j in pointer:
                                    temp_file.append(auth_list_stemmed[j])

                            temp_file = tfidf.fit_transform(temp_file)
                            cosine_output = cosine_similarity(temp_file, tfidf.transform(stem_word_file))

                            for j in pointer:
                                search_results[j] = cosine_output[pointer.index(j)]

                aa = 0
                rank_sorting = sorted(search_results.items(), key=lambda z: z[1], reverse=True)
                for a in rank_sorting:
                    outputField.insert(INSERT, "Rank: ")
                    outputField.insert(INSERT, "{:.2f}".format(a[1][0])) #Print rank
                    outputField.insert(INSERT, "\n")
                    outputField.insert(INSERT, 'Title: ' + pub_name[a[0]] + "\n") #Print Title of publication
                    outputField.insert(INSERT, 'URL: ' + pub_url[a[0]] + "\n") #Print URL
                    outputField.insert(INSERT, 'Date: ' + pub_date[a[0]] + "\n") #Print Date
                    outputField.insert(INSERT, 'Cov_Uni_Author: ' + pub_cu_author[a[0]] + "\n") #Print Author name
                    outputField.insert(INSERT, "\n")

                    aa = aa + 1
                if aa == 0:
                    messagebox.showinfo("Error!!!", "No results found. TRY AGAIN!!!")
                outputCount.insert(END, aa)

                exec_time = format((time.time() - start_time), ".2f")
                queryTimeEntry.insert(END, exec_time + " s")
            else:
                messagebox.showinfo("Invalid Input!!!", "Please enter more than 3 characters.")


# GUI build steps
window = Tk()
window.title("150265-STW7071CEM (Information Retrieval)")
window.configure(bg='#4775d1')
window.geometry('1000x750')

style = ttk.Style(window)
style.configure("TRadiobutton", background="#4775d1", foreground="white", font=("arial", 9))

llbHeading = Label(window, text="Vertical Search Engine", bg="#4775d1", fg="white", font="'Arial' 24 bold")
llbHeading.config(anchor=CENTER)
llbHeading.pack()

inputField = Entry(window, width=50)
inputField.pack()
inputField.place(x=20, y=72.5)

btnSearch = Button(window, text='SEARCH', bg="#788187", fg="#263543", font="'Arial' 10", command=search_and_publish)
btnSearch.place(x=340, y=70)

lblRule = Label(window, text='RULE :', font="'Arial' 9", bg="#4775d1", fg="white")
lblRule.place(x=810, y=70)

rule = IntVar()
rule.set(2)
rb_and = ttk.Radiobutton(window, text='AND', value=1, variable=rule, command=search_and_publish)
rb_and.pack(anchor=W)
rb_or = ttk.Radiobutton(window, text='OR', value=2, variable=rule, command=search_and_publish)
rb_or.pack(anchor=W)
rb_and.place(x=870, y=70)
rb_or.place(x=930, y=70)

lblResults = Label(window, text='Results : ', font="'Arial' 9", bg="#4775d1", fg="white")
lblResults.place(x=20, y=98.5)

outputCount = Entry(window, width=10, bg="#4775d1", fg="white", borderwidth=0)
outputCount.place(x=80, y=100)

lblQueryTime = Label(window, text='Query Time : ', font="'Arial' 9", bg="#4775d1", fg="white")
lblQueryTime.place(x=150, y=98.5)

queryTimeEntry = Entry(window, width=10, bg="#4775d1", fg="white", borderwidth=0)
queryTimeEntry.place(x=300, y=100)

outputField = scrolledtext.ScrolledText(window, width=117, height=37)
outputField.place(x=20, y=125)

window.bind('<Return>', search_and_publish)
window.mainloop()
