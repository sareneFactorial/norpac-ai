from ctypes import windll

import customtkinter as ctk

class App(ctk.CTk):


    # this will probably be a dict or something
    loaded_agents = ["asdf", "qwerty", "Test"]

    def __init__(self):
        super().__init__()

        # windll.shcore.SetProcessDpiAwareness(2)

        self.title("NorPac Bot")
        self.geometry("1600x900")

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_propagate(False)


        tabview = ctk.CTkTabview(master=self, anchor="se", corner_radius=0)
        tabview.grid(row=0, column=0, sticky="nsew")

        tabview.grid_columnconfigure(0, weight=1)
        tabview.grid_rowconfigure(0, weight=1)

        tabview.add("Edit")
        tabview.add("Stats")
        tabview.add("Train")
        tabview.add("Play")

        # # button = customtkinter.CTkButton(master=tabview.tab("Model"))
        # # button.pack(padx=20, pady=20)
        #
        edit_page = EditPage(master=tabview.tab("Edit"))
        edit_page.pack(expand=True, fill="both")
        #
        # a = ctk.CTkComboBox(master=tabview.tab("Pool"))
        # a.pack()

    def addAgent(self):
        """ Adds an agent to the pool. """
        pass

class EditPage(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)


        sidebar = ModelsSidebar(self)
        sidebar.grid(row=0, column=0, sticky="nws", padx=5, pady=5)

        tabview = ctk.CTkTabview(master=self, anchor="nw", corner_radius=0)
        tabview.grid(row=0, column=1, sticky="nsew")

        tabview.add("test")


        # self.optionmenu_var = ctk.StringVar(value="AlphaZero")
        # optionmenu = ctk.CTkOptionMenu(self, values=["Dueling DQN", "AlphaZero", "Random"], variable=self.optionmenu_var)
        # optionmenu.grid(row=0, column=1, padx=10)
        #
        # new_button = ctk.CTkButton(self, text="New Agent")
        # new_button.grid(row=0, column=2, padx=10)
        #
        # load_button = ctk.CTkButton(self, text="Load Agent")
        # load_button.grid(row=0, column=3, padx=10)

class ModelsSidebar(ctk.CTkScrollableFrame):
    """ Sidebar list of loaded models, like a file selector. """
    def __init__(self, master):
        super().__init__(master, width=200, corner_radius=0, fg_color="#2B2B2B")#, fg_color="blue")

        self.grid_propagate(flag=False)

        a = ModelListing(self, text="asdf", load=True, delete=True)
        a.grid(row=0, column=0, sticky="nw")

        b = ModelListing(self, text="qwerty", load=True, delete=True)
        b.grid(row=1, column=0, sticky="nw")

class ModelListing(ctk.CTkFrame):

    HEIGHT = 30
    BUTTON_SIZE = 20
    def __init__(self, master, text="Test", load=False, delete=False, select=False):
        super().__init__(master, corner_radius=0, height=self.HEIGHT)

        self.grid_columnconfigure(0, weight=1)
        self.grid_propagate(False)

        a = ctk.CTkLabel(self, text=text)
        a.grid(row=0, column=0, sticky="w", padx=5)

        if load:
            l = ctk.CTkButton(self, text="L", width=self.BUTTON_SIZE, height=self.BUTTON_SIZE)
            l.grid(row=0, column=1, sticky="e", padx=2.5)
        if delete:
            l = ctk.CTkButton(self, text="X", width=self.BUTTON_SIZE, height=self.BUTTON_SIZE, fg_color="red")
            l.grid(row=0, column=2, sticky="e", padx=2.5)
        if select:
            l = ctk.CTkButton(self, text="S", width=self.BUTTON_SIZE, height=self.BUTTON_SIZE, fg_color="forest green")
            l.grid(row=0, column=3, sticky="e", padx=2.5)

class

app = App()
app.mainloop()