"""main.py"""
import tkinter as tk
from HairlineTrackerGUI import HairlineTrackerGUI

def main():
    """ Main method for the program"""
    root = tk.Tk()
    HairlineTrackerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
