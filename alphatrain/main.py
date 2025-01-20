import os
from tkinter import *
from tkinter import ttk
import sv_ttk

# TODO: make this non-linux-only
# checkpointsDir = os.path.join(os.path.dirname(__file__), "checkpoints/")


class TrainingUI:
    def __init__(self, root):
        self.learnRate = StringVar(value="6e-5")
        self.valueBufferSize = StringVar(value="20000")
        self.policyBufferSize = StringVar(value="10000")

        self.eeConstant = StringVar(value="1.0")
        self.numSims = StringVar(value="300")

        self.gamesPerGen = StringVar(value="1")
        self.numGens = StringVar(value="1")
        self.checkpointFreq = StringVar(value="10")

        self.checkpointDir = StringVar(value="checkpoints/")

        self.loadCheckpoint = StringVar()


        n = ttk.Notebook(root)
        n.grid(column=0, row=0, sticky=(N, W, E, S))
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        self.trainingFrame = ttk.Frame(n)
        self.statsFrame = ttk.Frame(n)
        self.gameFrame = ttk.Frame(n)
        n.add(self.trainingFrame, text="Training")
        n.add(self.statsFrame, text="Stats")
        n.add(self.gameFrame, text="Game")

        self.trainingPage(self.trainingFrame)

    def trainingPage(self, f):
        t = Text(f, width=40, height=10)
        t.grid(row=0, column=0, rowspan=3, sticky="news", padx=5, pady=5)


        frame = ttk.Frame(f)
        loadFrame = ttk.Frame(frame)
        self.textEntry(loadFrame, 0, 0, self.loadCheckpoint, "Load Checkpoint (Relative)")
        b = ttk.Button(loadFrame, text="Load")
        b.grid(row=0, column=2)
        loadFrame.grid(row=0, column=1, sticky="news")
        for child in loadFrame.winfo_children():
            child.grid_configure(padx=5, pady=5)

        lf1 = ttk.Labelframe(frame, text="Training Settings")
        self.textEntry(lf1, 0, 1, self.learnRate, "Learn Rate")
        self.textEntry(lf1, 1, 1, self.valueBufferSize, "Value Buffer Size")
        self.textEntry(lf1, 2, 1, self.policyBufferSize, "Policy Buffer Size")
        lf1.grid(row=1, column=1, sticky="nsew")
        lf1.grid_columnconfigure(1, weight=1)
        for child in lf1.winfo_children():
            child.grid_configure(padx=10, pady=5)

        lf2 = ttk.Labelframe(frame, text="Generation Settings")
        self.textEntry(lf2, 0, 1, self.gamesPerGen, "Games Per Generation")
        self.textEntry(lf2, 1, 1, self.numGens, "Number of Generations")
        self.textEntry(lf2, 2, 1, self.checkpointFreq, "Gens Per Checkpoint")
        self.textEntry(lf2, 3, 1, self.checkpointDir, "Checkpoint Dir (Relative)")
        lf2.grid(row=2, column=1, sticky="nsew")
        lf2.grid_columnconfigure(1, weight=1)
        for child in lf2.winfo_children():
            child.grid_configure(padx=10, pady=5)

        lf3 = ttk.Labelframe(frame, text="MCTS Settings")
        self.textEntry(lf3, 0, 1, self.eeConstant, "EE Constant")
        self.textEntry(lf3, 1, 1, self.numSims, "Number of Simulations")
        lf3.grid(row=3, column=1, sticky="nsew")
        lf3.grid_columnconfigure(1, weight=1)
        for child in lf3.winfo_children():
            child.grid_configure(padx=10, pady=5)

        frame.grid_columnconfigure(1, weight=1)

        for child in frame.winfo_children():
            child.grid_configure(padx=5, pady=10)
        loadFrame.grid_configure(padx=5, pady=(5, 0))
        lf1.grid_configure(pady=(0, 5))


        f.grid_columnconfigure(0, weight=1)
        frame.grid(row=1, column=1, sticky=E)

    def textEntry(self, frame, row, column, var, text):
        e = ttk.Entry(frame, textvariable=var, width=20)
        e.grid(row=row, column=column+1, sticky=E)
        l = ttk.Label(frame, text=text)
        l.grid(row=row, column=column, sticky=W)



if __name__ == '__main__':
    window = Tk()

    t = TrainingUI(window)

    sv_ttk.set_theme("dark")

    window.mainloop()
    # test()

# cProfile.run("test()", sort='cumtime')
