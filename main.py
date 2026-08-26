import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import random
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ------------------------- CSV Utilities -------------------------
def load_csv_file(path):
    df = pd.read_csv(path, parse_dates=['date'])
    df = df[['date', 'meantemp', 'humidity', 'wind_speed', 'meanpressure']].dropna()
    df['year'] = df['date'].dt.year
    return df

def filter_year_range(df, start_year, end_year):
    return df[(df['year'] >= start_year) & (df['year'] <= end_year)].reset_index(drop=True)

# ------------------------- PCA Utilities -------------------------
def standardize_matrix(X):
    rows = len(X)
    cols = len(X[0])

    # Step 1: Compute column-wise mean
    mean = []
    for c in range(cols):
        col_sum = sum(X[r][c] for r in range(rows))
        mean.append(col_sum / rows)

    # Step 2: Compute standard deviation
    std = []
    for c in range(cols):
        variance_sum = sum((X[r][c] - mean[c]) ** 2 for r in range(rows))
        variance = variance_sum / (rows - 1)
        std_value = variance ** 0.5 if variance > 0 else 1.0
        std.append(std_value)

    # Step 3: Standardize each value
    Xs = [[(X[r][c] - mean[c]) / std[c] for c in range(cols)] for r in range(rows)]
    return Xs, mean, std

def transpose(A):
    return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]

def matmul(A, B):
    rows_A, cols_A = len(A), len(A[0])
    cols_B = len(B[0])
    result = []
    for i in range(rows_A):
        new_row = []
        for j in range(cols_B):
            total = sum(A[i][k] * B[k][j] for k in range(cols_A))
            new_row.append(total)
        result.append(new_row)
    return result

def matvec(A, v):
    return [sum(A[i][j] * v[j] for j in range(len(A[0]))) for i in range(len(A))]

def vector_norm(v):
    return sum(x**2 for x in v) ** 0.5

def power_iteration(A, max_iter=1000, tol=1e-9):
    n = len(A)
    b = [random.random() for _ in range(n)]

    for _ in range(max_iter):
        b_new = [sum(A[i][j] * b[j] for j in range(n)) for i in range(n)]
        norm_b = vector_norm(b_new)
        b_new = [x / norm_b for x in b_new]
        if sum(abs(b_new[i] - b[i]) for i in range(n)) < tol:
            break
        b = b_new

    Ab = [sum(A[i][j] * b[j] for j in range(n)) for i in range(n)]
    eigenvalue = sum(Ab[i] * b[i] for i in range(n))
    return eigenvalue, b

def manual_pca(X):
    rows, cols = len(X), len(X[0])

    # Center the matrix
    X_centered = []
    for r in range(rows):
        new_row = []
        for c in range(cols):
            mean_c = sum(row[c] for row in X) / rows
            new_row.append(X[r][c] - mean_c)
        X_centered.append(new_row)

    # Covariance matrix
    XT = transpose(X_centered)
    cov = [[sum(XT[i][k] * X_centered[k][j] for k in range(rows)) / (rows - 1)
            for j in range(cols)] for i in range(cols)]

    n = len(cov)
    eigvals, eigvecs = [], []
    A = [row[:] for row in cov]

    for _ in range(n):
        val, vec = power_iteration(A)
        eigvals.append(val)
        eigvecs.append(vec)

        outer = [[vec[i] * vec[j] * val for j in range(n)] for i in range(n)]
        A = [[A[i][j] - outer[i][j] for j in range(n)] for i in range(n)]

    idx = sorted(range(n), key=lambda i: eigvals[i], reverse=True)
    sorted_eigvals = [eigvals[i] for i in idx]
    sorted_eigvecs = [[eigvecs[j][i] for j in range(n)] for i in idx]
    eigvecs_T = transpose(sorted_eigvecs)

    PC = [[sum(X_centered[r][c] * eigvecs_T[c][k] for c in range(n)) for k in range(n)]
          for r in range(rows)]

    return PC, sorted_eigvals, eigvecs_T

# ------------------------- GUI Class -------------------------
class PCAWeatherGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Weather PCA - Fully Manual Linear Algebra")
        self.root.geometry("1200x700")
        self.csv_path = None
        self.start_year_var = tk.IntVar(value=2009)
        self.end_year_var = tk.IntVar(value=2010)
        self.setup_ui()

    def setup_ui(self):
        # Left frame for controls
        ctrl_frame = ttk.Frame(self.root, padding=10)
        ctrl_frame.grid(row=0, column=0, sticky='nw')
        ctrl_frame.grid_rowconfigure(10, weight=1)

        ttk.Button(ctrl_frame, text="Load CSV", command=self.browse_file).grid(row=0, column=0, columnspan=2, pady=5)
        ttk.Label(ctrl_frame, text="Start Year:").grid(row=1, column=0, sticky='e', pady=2)
        ttk.Entry(ctrl_frame, textvariable=self.start_year_var, width=6).grid(row=1, column=1, sticky='w')
        ttk.Label(ctrl_frame, text="End Year:").grid(row=2, column=0, sticky='e', pady=2)
        ttk.Entry(ctrl_frame, textvariable=self.end_year_var, width=6).grid(row=2, column=1, sticky='w')
        ttk.Button(ctrl_frame, text="Run PCA & Visualise", command=self.run_analysis).grid(row=3, column=0, columnspan=2, pady=8)

        # Right frame for plots
        self.tab_control = ttk.Notebook(self.root)
        self.tab_control.grid(row=0, column=1, sticky='nsew')
        self.plot_tabs = {}
        for tab_name in ["Original Features", "Scree Plot", "2D PCA Projection"]:
            tab = ttk.Frame(self.tab_control)
            self.plot_tabs[tab_name] = tab
            self.tab_control.add(tab, text=tab_name)

        # Bottom frame for interpretation
        self.text_frame = ttk.Frame(self.root, padding=5)
        self.text_frame.grid(row=1, column=0, columnspan=2, sticky='nsew')
        self.text_frame.grid_columnconfigure(0, weight=1)
        self.interpretation = tk.Text(self.text_frame, height=12, width=140)
        self.interpretation.pack(side='left', fill='both', expand=1)
        scrollbar = ttk.Scrollbar(self.text_frame, command=self.interpretation.yview)
        scrollbar.pack(side='right', fill='y')
        self.interpretation.config(yscrollcommand=scrollbar.set)

        # Configure resizing
        self.root.grid_rowconfigure(0, weight=3)
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

    def browse_file(self):
        path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if path:
            self.csv_path = path
            messagebox.showinfo("File Selected", f"Selected file: {path}")

    def run_analysis(self):
        if not self.csv_path:
            messagebox.showerror("Error", "Please load a CSV file first.")
            return
        sy, ey = self.start_year_var.get(), self.end_year_var.get()
        if sy > ey:
            messagebox.showerror("Error", "Start year must be <= End year")
            return

        df = load_csv_file(self.csv_path)
        df_f = filter_year_range(df, sy, ey)
        if df_f.empty:
            messagebox.showerror("Error", "No data in the selected year range.")
            return

        features = ['meantemp', 'humidity', 'wind_speed', 'meanpressure']
        X_raw = df_f[features].values.tolist()
        PC, eigvals, eigvecs = manual_pca(X_raw)
        self.show_plot_original(df_f, features)
        self.show_plot_scree(eigvals)
        self.show_plot_projection(df_f, PC)
        self.show_interpretation(eigvals, eigvecs, features)

    def clear_tab(self, tab_name):
        for widget in self.plot_tabs[tab_name].winfo_children():
            widget.destroy()

    def show_plot_original(self, df, features):
        tab_name = "Original Features"
        self.clear_tab(tab_name)
        fig = Figure(figsize=(8, 5), dpi=100)
        ax = fig.add_subplot(111)
        df.set_index('date')[features].plot(ax=ax)
        ax.set_title("Original Weather Features")
        ax.set_ylabel("Value")
        canvas = FigureCanvasTkAgg(fig, master=self.plot_tabs[tab_name])
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=1)

    def show_plot_scree(self, eigvals):
        tab_name = "Scree Plot"
        self.clear_tab(tab_name)
        fig = Figure(figsize=(6, 5), dpi=100)
        ax = fig.add_subplot(111)
        explained_var_ratio = [v / sum(eigvals) * 100 for v in eigvals]
        pcs = list(range(1, len(eigvals) + 1))
        ax.bar(pcs, explained_var_ratio, color='teal')
        ax.set_xlabel("Principal Component")
        ax.set_ylabel("Variance Explained (%)")
        ax.set_title("Scree Plot")
        canvas = FigureCanvasTkAgg(fig, master=self.plot_tabs[tab_name])
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=1)

    def show_plot_projection(self, df, PC):
        tab_name = "2D PCA Projection"
        self.clear_tab(tab_name)
        fig = Figure(figsize=(8, 5), dpi=100)
        ax = fig.add_subplot(111)
        dates = df['date']
        ax.plot(dates, [row[0] for row in PC], label='PC1', color='blue')
        ax.plot(dates, [row[1] for row in PC], label='PC2', color='orange')
        ax.set_title("PCA (2 Components)")
        ax.set_ylabel("PC Values")
        ax.set_xlabel("Date")
        ax.legend()
        ax.grid(True)
        canvas = FigureCanvasTkAgg(fig, master=self.plot_tabs[tab_name])
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=1)

    def show_interpretation(self, eigvals, eigvecs, features):
        text = "--- PCA Interpretation ---\n"
        total_var = sum(eigvals)
        for i in range(2):
            text += f"PC{i+1} explains {eigvals[i]/total_var*100:.2f}% of variance.\n"
        text += "\nPrincipal Component Loadings:\n"
        for i in range(2):
            text += f"PC{i+1}: " + ", ".join(
                [f"{features[j]}={eigvecs[j][i]:.2f}" for j in range(len(features))]
            ) + "\n"
        text += "\nInsights:\n- PC1 captures overall trend (largest variance).\n- PC2 captures secondary variations.\n"
        self.interpretation.delete('1.0', tk.END)
        self.interpretation.insert(tk.END, text)

# ------------------------- Main Runner -------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = PCAWeatherGUI(root)
    root.mainloop()