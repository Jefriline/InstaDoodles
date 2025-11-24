import tkinter as tk

try:
    import customtkinter as ctk
    _USE_CTK = True
except ImportError:
    _USE_CTK = False

from app.ui import InstaDoodlesApp


def main() -> None:
    root = tk.Tk()
    root.title("InstaDoodles - Alinear y Dibujar")
    
    app = InstaDoodlesApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
