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

#from pages import 1_Visual Database, 2_Gene Detection, 3_Make Predictions

import tensorflow
plt.style.use('default')

st.set_page_config(
    page_title = 'Slide Gene Detection V1.0',   #
    #page_icon = '🕵️‍♀️',
    layout = 'wide'
)

st.header('📑使用说明')
st.write('1. 软件启动。在电脑终端指定文件夹下输入：streamlit run Home.py')
# st.write('2. 首先进行变量筛选，通过变量筛选，将分析的重点集中在结局变量相关的特征上，有助于增加分析的速度和效率，避免在不必要的特征上花费时间和算力的浪费；\
#     然后进行SHAP分析，SHAP值的计算需要大量的运算，过多的变量会导致会导致计算时间较长，也体现了变量筛选的必要性；\
#     最后进行曲线拟合，通过将散点图拟合到曲线上，可以确定关键点的坐标，进而说明变量之间的关系。')
'***'
st.header('🎯用途说明')
st.write('1. 矿区可视化数据库。司家营矿区边坡滑坡的重要特征参数数字化展示及特征统计分布。')
st.write('2. 矿区边坡致滑特征基因诊断。计算司家营矿区全局基因特征重要性。进行矿区特征基因全局诊断和个体诊断。')
st.write('3. 单体边坡滑坡基因预测。预测生产中可能产生的新边坡的滑坡危险性。')

'***'
st.header('🔧制作说明')
st.write('1. 评价单元采用基于分水岭与沟谷线划分的斜坡单元。')
st.write('2. 地形地貌数据基于2023年矿区无人机倾斜摄影测量数据计算而得；地质数据、人类工程数据基于资料收集、现场地质调查；降雨数据基于气象台站；地表位移速率基于哨兵1号合成孔径雷达测量。')
st.write('3. 数据重采样采用SMOTENC-TomekLinks算法；变量筛选采用LightGBM算法；模型解释采用SHAP算法。')
st.write('4. 开发软件版本：Python_3.11.6。')
st.write('5. 制作单位：河北钢铁集团滦县司家营铁矿有限公司，北京科技大学。')



placeholder = st.empty()
with placeholder.container():
    f1 = st.columns(1)
    with f1[0]:
        img = Image.open("hegang_bk2.tif")
        st.image(img)

