import streamlit.web.cli as stcli
import os, sys
from Home import *
import time
from PIL import Image
import streamlit as st
from streamlit_shap import st_shap
import shap
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MaxNLocator
import plotly.express as px
import seaborn as sns
import plotly.figure_factory as ff
import geopandas as gpd
import joblib


import time
from PIL import Image
import streamlit as st
from streamlit_shap import st_shap
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import plotly.express as px
import seaborn as sns
import plotly.figure_factory as ff
import joblib
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer
from plotly import graph_objects as go
from lightgbm import plot_importance
import geopandas as gpd

import time
from PIL import Image
import streamlit as st
from streamlit_shap import st_shap
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import plotly.express as px
import seaborn as sns
import plotly.figure_factory as ff
import joblib
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer
from plotly import graph_objects as go
import geopandas as gpd

import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import time
from streamlit_shap import st_shap
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MaxNLocator
import plotly.express as px
import seaborn as sns
import plotly.figure_factory as ff
import joblib
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer
from plotly import graph_objects as go
import lightgbm as lgb
from lightgbm import plot_importance
import geopandas as gpd
import streamlit as st




if __name__ == "__main__":
    # 这里的 streamlit app 是什么名字，需要修改下
    sys.argv = [
        "streamlit",
        "run",
        "Home.py",  # 这里
        "--global.developmentMode=false",
    ]
    sys.exit(stcli.main())

