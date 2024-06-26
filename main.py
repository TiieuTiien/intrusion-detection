import os
import csv
import joblib
from tkinter import filedialog
from gui import App
from intrusionDetection import evaluate, evaluate_all
from time import perf_counter

class MainApp(App):
    def __init__(self):
        super().__init__()
        self.dataset = None
        self.models = {}
        
        # Set default paths
        self.default_dataset_path = 'dataset/kddcup1999.csv'
        self.default_models_folder = 'models/'

        # Load default dataset and models
        self.load_default_dataset()
        self.load_default_models()

        # Override button commands after GUI initialization
        self.button_load_data.config(command=self.open_dataset)
        self.button_models_path.config(command=self.open_models)
        self.button_train.config(command=self.train)
        self.button_eval_all.config(command=self.eval_all)

    def load_default_dataset(self):
        if os.path.exists(self.default_dataset_path):
            self.dataset = self.default_dataset_path
            self.display_csv_file_content(self.dataset)
            file_name = os.path.basename(self.dataset)  # Extract file name
            self.label_train_data.config(text=file_name)
        else:
            print(f"Default dataset path {self.default_dataset_path} does not exist.")

    def load_default_models(self):
        if os.path.exists(self.default_models_folder):
            default_models_folder = self.default_models_folder + os.path.basename(self.dataset)[:-4]
            self.label_models_path.config(text=default_models_folder)
            model_files = [f for f in os.listdir(default_models_folder) if f.endswith('.pkl')]
            self.models = {}
            menu = self.model_option_menu["menu"]
            menu.delete(0)

            for model_file in model_files:
                model_path = os.path.join(default_models_folder, model_file)
                model_name = os.path.splitext(model_file)[0]
                self.models[model_name] = joblib.load(model_path)
                print(f"Model {model_name} loaded successfully")
                # Add new options to the OptionMenu
                menu.add_command(label=model_name, command=lambda value=model_name: self.selected_model.set(value))

            # Set the default value to the first option
            if model_files:
                self.selected_model.set(os.path.splitext(model_files[0])[0])
            else:
                self.selected_model.set("None")
        else:
            print(f"Default models folder {default_models_folder} does not exist.")

    def open_dataset(self):
        self.dataset = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if self.dataset:
            file_name = os.path.basename(self.dataset)  # Extract file name
            self.label_train_data.config(text=file_name)
            self.display_csv_file_content(self.dataset)

            models_folder = self.default_models_folder + os.path.basename(self.dataset)[:-4]
            print(models_folder)
            if (models_folder):
                self.label_models_path.config(os.path.basename(self.default_models_folder))
                model_files = [f for f in os.listdir(models_folder) if f.endswith('.pkl')]
                self.models = {}
                menu = self.model_option_menu["menu"]
                menu.delete(0)

                for model_file in model_files:
                    model_path = os.path.join(models_folder, model_file)
                    model_name = os.path.splitext(model_file)[0]
                    self.models[model_name] = joblib.load(model_path)
                    print(f"Model {model_name} loaded successfully")
                    # Add new options to the OptionMenu
                    menu.add_command(label=model_name, command=lambda value=model_name: self.selected_model.set(value))

                    # Set the default value to the first option
                if model_files:
                    self.selected_model.set(os.path.splitext(model_files[0])[0])
                else:
                    self.selected_model.set("None")

    def open_models(self):
        # Prompt the user to select a folder
        models_folder = filedialog.askdirectory()
        models_folder = models_folder + '/' + os.path.basename(self.dataset)[:-4]

        if (models_folder):
            self.label_models_path.config(os.path.basename(self.default_models_folder))
            model_files = [f for f in os.listdir(models_folder) if f.endswith('.pkl')]
            self.models = {}
            menu = self.model_option_menu["menu"]
            menu.delete(0)

            for model_file in model_files:
                model_path = os.path.join(models_folder, model_file)
                model_name = os.path.splitext(model_file)[0]
                self.models[model_name] = joblib.load(model_path)
                print(f"Model {model_name} loaded successfully")
                # Add new options to the OptionMenu
                menu.add_command(label=model_name, command=lambda value=model_name: self.selected_model.set(value))

                # Set the default value to the first option
            if model_files:
                self.selected_model.set(os.path.splitext(model_files[0])[0])
            else:
                self.selected_model.set("None")

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
        evaluate(self.models[self.selected_model.get()], self.dataset)

    def eval_all(self):
        print(list(self.models.values()))
        evaluate_all(list(self.models.values()), self.dataset)

if __name__ == "__main__":
    app = MainApp()
    app.mainloop()