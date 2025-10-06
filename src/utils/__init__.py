# ==============================================================
# 🔍 COMPROBACIÓN DEL ENTORNO DE TRABAJO
# ==============================================================

import sys
import numpy as np
import pandas as pd
import sklearn
import matplotlib
import seaborn
import openml
import joblib

print("✅ Entorno de trabajo correctamente configurado.\n")
print(f"Versión de Python: {sys.version.split()[0]}")
print(f"Numpy: {np.__version__}")
print(f"Pandas: {pd.__version__}")
print(f"Scikit-learn: {sklearn.__version__}")
print(f"Matplotlib: {matplotlib.__version__}")
print(f"Seaborn: {seaborn.__version__}")
print(f"OpenML: {openml.__version__}")
print(f"Joblib: {joblib.__version__}")
