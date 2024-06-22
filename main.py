import os
import csv
from tkinter import filedialog
from gui import App
from intrusionDetection import training
from time import perf_counter

class MainApp(App):
    def __init__(self):
        super().__init__()
        self.train_file = None
        self.test_file = None

        # Override button commands after GUI initialization
        self.button_train_csv.config(command=self.open_file_train)
        self.button_test_csv.config(command=self.open_file_test)
        self.button_train.config(command=self.train)

    def open_file_train(self):
        self.train_file = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if self.train_file:
            file_name = os.path.basename(self.train_file)  # Extract file name
            self.label_train_csv.config(text=file_name)
            self.display_csv_file_content(self.train_file)

    def open_file_test(self):
        self.test_file = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if self.test_file:
            file_name = os.path.basename(self.test_file)  # Extract file names
            self.label_test_csv.config(text=file_name)

    def display_csv_file_content(self, file_path):
        self.textbox1.delete("1.0", "end")

        with open(file_path, "r", newline="") as file:
            csv_reader = csv.reader(file)
            header = next(csv_reader)
            header_text = ", ".join(header)
            self.textbox1.insert("end", "This is header of file \n\n\n")
            self.textbox1.insert("end", header_text)

    def train(self):
        t1 = perf_counter()
        print("Loading ....")
        self.textbox2.delete("1.0", "end")
        self.textbox2.insert("1.0", "Loading....\n")
        print(self.selected_model.get())

        sensitivity, specificity, y_test, predictions = training(self.selected_model.get(), self.train_file, self.test_file)
        correct_count = (y_test == predictions).sum()
        incorrect_count = (y_test != predictions).sum()
        t2 = perf_counter()
        self.textbox2.insert("end", f"This is result of: {self.selected_model.get()} model \n\n")
        self.textbox2.insert("end", f"Correct: {correct_count}\n")
        self.textbox2.insert("end", f"Incorrect: {incorrect_count}\n")
        self.textbox2.insert("end", f"True Positive Rate: {100 * sensitivity:.2f}%\n")
        self.textbox2.insert("end", f"False Negative Rate: {100 * (1- specificity):.2f}%\n")
        self.textbox2.insert("end", f"Time: {t2-t1:.2f}s\n")
        print(f"The time to train and test: {t2-t1}\n")

if __name__ == "__main__":
    app = MainApp()
    app.mainloop()
