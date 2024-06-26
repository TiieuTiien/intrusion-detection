import ttkbootstrap as ttk
import pandas as pd
import tkinter as tk
from ttkbootstrap.constants import *
from tkinter.scrolledtext import ScrolledText
from tkinter import StringVar, OptionMenu, filedialog

class App(ttk.Window):
    def __init__(self):
        super().__init__()

        self.train_file = None
        self.test_file = None

        # Set initial window size (optional)
        self.geometry("900x600")  # Start with screen size, adjust as needed
        self.title("Nhóm 11 - Intrusion Detection")
        self.resizable(True, True)
        self.place_window_center()

        # Create and configure main window grid
        self.grid_rowconfigure((0,1,2,4,5), weight=1)  # Expand row to fill available space
        self.grid_rowconfigure((3,6), weight=4)  # Expand row to fill available space
        self.grid_columnconfigure((0,3), weight=1)  # Expand column to fill available space
        self.grid_columnconfigure((1,2), weight=3)  # Expand column to fill available space

        self.creWindow()  # Pass content_frame to creWindow

    def creWindow(self):

        ttk.Label(self, text="Intrusion detection", font="Arial 15 bold", foreground="#333333").grid(row=0, column=1, columnspan=2, pady=10)

        # Create widgets directly on the content_frame
        import_data_frame = ttk.Frame(self)
        import_data_frame.grid(row=1, column=1, sticky="nsew")
        import_data_frame.columnconfigure((0,1,2,3,4), weight=1)

        ttk.Label(import_data_frame, text="Datasets:").grid(row=0, column=0, sticky="e")
        
        self.button_load_data = ttk.Button(import_data_frame, text="Dataset", width=9, padding=8)
        self.button_load_data.grid(row=0, column=1)
        self.label_train_data = ttk.Label(import_data_frame, text="")
        self.label_train_data.grid(row=0, column=2)

        ttk.Label(import_data_frame, text="Models path:").grid(row=0, column=3, sticky="e")

        self.button_models_path = ttk.Button(import_data_frame, text="Model path", width=10, padding=8)
        self.button_models_path.grid(row=0, column=4)

        # Model Selection
        select_datasets = ttk.Frame(self)
        select_datasets.grid(row=1, column=2, sticky="nsew")
        select_datasets.columnconfigure((0,1), weight=1)

        ttk.Label(select_datasets, text="Model:").grid(row=0, column=0, sticky="e")
        # options = ["RandomForestClassifier", "LogisticRegression", "MLPClassifier", "DecisionTreeClassifier"]
        self.selected_model = StringVar(self)
        self.selected_model.set("None")
        self.model_option_menu = OptionMenu(select_datasets, self.selected_model, "None")
        self.model_option_menu.grid(row=0, column=1, sticky="w")

        ttk.Label(self, text="Table header:").grid(row=2, column=1, pady=10, sticky="w")
        self.textbox1 = ScrolledText(self, width=80, height=8, font="Arial 10 bold")
        self.textbox1.grid(row=3, column=1, columnspan=2, sticky="nsew")

        self.button_train = ttk.Button(self, text="Training", style="warning", width=9, padding=8, cursor="hand2")
        self.button_train.grid(row=4, column=1)
        
        self.button_eval_all = ttk.Button(self, text="Evaluate All", style="warning", width=9, padding=8, cursor="hand2")
        self.button_eval_all.grid(row=4, column=2)

        ttk.Label(self, text="Accuracy:").grid(row=5, column=1, pady=10, sticky="w")
        self.textbox2 = ScrolledText(self, width=80, height=8, font="Arial 10 bold")
        self.textbox2.grid(row=6, column=1, columnspan=2, sticky="nsew")

# if __name__ == "__main__":
#     app = App()
#     app.mainloop()