import matplotlib
from matplotlib.backends.backend_pgf import FigureCanvasPgf

# Always process .pdf files with pgf backend
matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)

# Set LaTeX parameters
matplotlib.rcParams.update({
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})
