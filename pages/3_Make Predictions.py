# -*- coding: utf-8 -*-
"""
@author: Jiang Li
"""

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
shap.initjs()


# 设置 Matplotlib 中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
plt.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'Arial']

plt.style.use('default')

st.set_page_config(
    page_title = 'Slide Gene Detection V1.0',
    #page_icon = '🕵️‍♀️',
    layout = 'wide'
)

# dashboard title
st.markdown("<h1 style='text-align: center; color: black;'>矿区边坡滑坡基因诊断</h1>", unsafe_allow_html=True)
''
''

# 数据读取，建立变量
sjy = gpd.read_file(r"SJY_z_slide_v.geojson")
X_vars = ['height','slope','profile curvature','lithology','rock texture','d_river','d_road','velocity']
x = sjy[X_vars]
y = sjy['slide']
x_copy = sjy[X_vars].copy()

# 创建一个字典来存储每个列的LabelEncoder实例和映射
encoders = {}
mappings = {}

# 列表中的列需要编码
columns_to_encode = ['lithology', 'rock texture']

for col in columns_to_encode:
    le = LabelEncoder()
    x_copy.loc[:, col] = le.fit_transform(x_copy[col])
    x_copy[col] = x_copy[col].astype('int64')
    encoders[col] = le
    mappings[col] = dict(zip(le.classes_, le.transform(le.classes_)))

# 计算新的变量
x_copy['velocity^2'] = x_copy['velocity'] ** 2
x_copy['height slope'] = x_copy['height'] * x_copy['slope']
x_copy['d_river^2'] = x_copy['d_river'] ** 2
x_copy['profile curvature^2'] = x_copy['profile curvature'] ** 2
x_copy['lithology d_river'] = x_copy['lithology'] * x_copy['d_river']
x_copy['d_road^2'] = x_copy['d_road'] ** 2

# 选择原始变量和新变量组成新的 DataFrame
x_copy_poly = x_copy[['d_river', 'height', 'profile curvature', 'rock texture',
                      'velocity^2', 'height slope', 'd_river^2', 'profile curvature^2',
                      'lithology d_river', 'd_road^2']]

# 加载模型
loaded_model = joblib.load('SJY_slide_yuce.pkl')
train_x3, test_x3, train_y3, test_y3 = train_test_split(x_copy_poly, y, test_size=0.2, random_state=42)

def user_input_features():
    st.sidebar.header('边坡特征输入')
    #st.sidebar.write('User input parameters below ⬇️')
    height = float(st.sidebar.number_input('坡高（m）', min_value=x_copy['height'].min(), max_value=x_copy['height'].max()))
    slope = float(st.sidebar.number_input('坡度（度）', min_value=x_copy['slope'].min(), max_value=x_copy['slope'].max()))
    profile_curvature = float(st.sidebar.number_input('剖面曲率：负凸，正凹', min_value=x_copy['profile curvature'].min(), max_value=x_copy['profile curvature'].max()))
    lithology = int(st.sidebar.selectbox('岩性', options=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']))
    st.sidebar.write(f'0-中粗含砾石英砂岩,1-中风化黑云变粒岩,2-变质辉长岩,3-底砾岩,4-强风化黑云变粒岩,5-微风化黑云变粒岩,6-混合化黑云变粒岩,7-燧石白云岩,8-白云岩,\
    9-石英砂岩,10-石英砂岩与白云岩互层,11-覆盖土砂层,12-None  ')
    rock_texture = int(st.sidebar.selectbox('岩石结构', options=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']))
    st.sidebar.write('0-中粒状变晶结构,1-中至粗粒结构,2-交代重结晶微晶结构,3-单粒结构,4-砂状结构,5-砾状结构,6-碎屑微晶结构,7-碎屑结构,8-粒状变晶结构,\
    9-细粒鳞片、粒状变晶结构,10-隐晶_自形粒状结构,11-None ')
    d_river = float(st.sidebar.number_input('距河流距离（m）', min_value=x_copy['d_river'].min(), max_value=x_copy['d_river'].max()))
    d_road = float(st.sidebar.number_input('距道路距离（m）', min_value=x_copy['d_road'].min(), max_value=x_copy['d_road'].max()))
    velocity = float(st.sidebar.number_input('地表位移变化速率（mm/年）：负沉，正升', min_value=x_copy['velocity'].min(),max_value=x_copy['velocity'].max()))


    output = [height, slope, profile_curvature, lithology, rock_texture, d_river, d_road, velocity]
    return output

# 调用函数并解包返回的值

output = user_input_features()
data = [output[5], output[0], output[2], output[4], output[7]**2, output[0]*output[1], output[5]**2, output[2]**2, output[3]*output[5], output[6]**2]
#将列表转换为 NumPy 数组，并确保它是二维的
data_array = np.array(data).reshape(1, -1)
#或者，如果您想使用 pandas DataFrame
data_df = pd.DataFrame(data_array, columns=['d_river', 'height', 'profile curvature', 'rock texture',
                      'velocity^2', 'height slope', 'd_river^2', 'profile curvature^2',
                     'lithology d_river', 'd_road^2'])  # 替换列名为实际的特征名
#构建数据表格
data_table = {
    '坡高（m）': output[0],
    '坡度（度）': output[1],
    '剖面曲率': output[2],
    '岩性': output[3],
    '岩石结构': output[4],
    '距河流距离（m）': output[5],
    '距道路距离（m）': output[6],
    '地表位移变化速率（mm/year）': output[7]
}


# 主页面
st.title('边坡稳定性预测')               #Make predictions in real time

if st.button('Predict'):  #.sidebar

    placeholder = st.empty()
    placeholder2 = st.empty()
    placeholder3 = st.empty()


    with placeholder.container():
        f1, f2 = st.columns(2)

        with f1:
            st.subheader('输入参数如下 ⬇️')  #User input parameters below
            st.table(data_table)
            '***'

        with f2:
            # 模拟风险概率计算
            risk_probability = loaded_model.predict_proba(data_array)[:, 1][0]
            risk_color = "green" if risk_probability < 0.25 else "green" if risk_probability < 0.50 else "orange" if risk_probability < 0.75 else "red"
            st.subheader('边坡滑坡危险性 ⬇️')
            st.markdown(f'<h1 style="color:{risk_color};">Risk Probability: {risk_probability:.2%}</h1>', unsafe_allow_html=True)
            ''
            ''
            ''
            ''
            ''
            st.markdown('''
            **预测边坡滑坡危险性等级划分**：  
            - *无危险性：Risk Probability<25%*  
            - *低危险性：25%<Risk Probability<50%*  
            - *中危险性：50%<Risk Probability<75%*  
            - *高危险性：Risk Probability>75%*          
            ''')
            '***'

    explainer = shap.TreeExplainer(loaded_model)
    shap_values = explainer.shap_values(data_df)
    shap_values2 = explainer(data_df)

    # 创建 Explanation 对象
    # explanation = shap.Explanation(
    #     values=shap_values[1], # 选择正类的 SHAP 值
    #     base_values=explainer.expected_value[1], # 正类的基线值
    #     data=data_df.iloc[0], # 输入数据
    #     feature_names=data_df.columns.tolist()  # 特征名称
    #)


    with placeholder2.container():
        f3= st.columns(1)

        with f3[0]:

            st.subheader('输入样本的特征基因链驱动力图')
            shap.force_plot(explainer.expected_value[1], shap_values[1][0, :], data_df.iloc[0, :], show=False,matplotlib=True)
            st.pyplot(plt.gcf())
            '***'


    with placeholder3.container():
        f4, f5= st.columns(2)

        with f4:

            st.markdown(
                '<div style="background-color:#4682B4;color:white;padding:10px;font-weight:bold;font-size:20px;">预测说明：</div>',
                unsafe_allow_html=True)
            st.markdown('''
                **取基因特征重要度排名前10组成特征基因链**：  
                - **地形地貌**：坡高、坡高*坡度互耦合、坡面剖面曲率、坡面剖面曲率自耦合；  
                - **地质水文**：岩石结构、距河流距离、距河流距离自耦合、岩性*距河流距离互耦合、地表位移变化速率；  
                - **人类工程**：距道路距离自耦合。    
            ''')
            st.markdown('''
                **样本特征基因链驱动力图**：  
                - 红色特征基因：致滑特征基因，长度决定致滑度；  
                - 蓝色特征基因：稳定特征基因。
            ''')

            '***'

        with f5:

            st.markdown(
                '<div style="background-color:#4682B4;color:white;padding:10px;font-weight:bold;font-size:20px;">防治建议：</div>',
                unsafe_allow_html=True)
            st.markdown('''             
            **各致滑特征基因对应危险、安全阈值**：  
            - **地形地貌**：坡高（<24m安全）；坡度（24°-55°安全）；剖面曲率（-10~1危险）；
            - **地质水文**：易滑岩性（1中/4强风化黑云变粒岩、8白云岩、9石英砂岩、11覆盖土砂层）；易滑岩石结构（0中粒状变晶结构、3单粒结构）；距河流距离（338m~675m危险）；地表位移变化速率（<0安全）；
            - **人类工程**：距道路距离（<10m危险）。  
            ''')
            st.markdown('*阈值范围由3节点3次方的限制性立方样条曲线拟合获得。*')
            st.markdown('**对影响力最大的致滑特征基因应着重整改，通过工程手段将其控制在安全区间。**')
            '***'