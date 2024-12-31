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

# 设置 Matplotlib 中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
plt.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'Arial']

plt.style.use('default')

st.set_page_config(
    page_title = 'Slide Gene Detection V1.0',   #
    #page_icon = '🕵️‍♀️',
    layout = 'wide'
)

# dashboard title
#st.title("Real-Time Fraud Detection Dashboard")
st.markdown("<h1 style='text-align: center; color: black;'>可视化数据库-司家营矿区</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: black;'>Visual Database For SJY</h1>", unsafe_allow_html=True)


# understand the dataset
sjy = gpd.read_file("SJY_z_slide_v.geojson")
X_vars = ['height','slope','profile curvature','lithology','rock texture','rock structure','d_fault','d_river','ppv','d_road','velocity','rfactor']
x = sjy[X_vars]
x_copy = sjy[X_vars].copy()
y = sjy['slide']

st.title('特征空间分布')
''
''

# 需要一个count plot
placeholder4= st.empty()
placeholder5 = st.empty()
placeholder6 = st.empty()

with placeholder4.container():
    f13,f14,f15,f16 = st.columns(4)

    with f13:
        plt.figure(figsize=(10, 10))
        plot = sjy.plot(column='height')  # 直接使用 plt 的当前活动图形
        plt.title("Height")
        plt.colorbar(plot.collections[0], label="Height Feature Value")
        plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=4))  # 使用 plt.gca() 获取当前轴

        # 使用 Streamlit 显示图形
        st.pyplot(plt)
    with f14:
        plt.figure(figsize=(10, 10))
        plot = sjy.plot(column='slope')  # 直接使用 plt 的当前活动图形
        plt.title("Slope")
        plt.colorbar(plot.collections[0], label="Slope Feature Value")
        plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=4))  # 使用 plt.gca() 获取当前轴

        # 使用 Streamlit 显示图形
        st.pyplot(plt)
    with f15:
        plt.figure(figsize=(10, 10))
        plot = sjy.plot(column='profile curvature')  # 直接使用 plt 的当前活动图形
        plt.title("Profile Curvature")
        plt.colorbar(plot.collections[0], label="Profile Curvature Feature Value")
        plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=4))  # 使用 plt.gca() 获取当前轴

        # 使用 Streamlit 显示图形
        st.pyplot(plt)
    with f16:
        plt.figure(figsize=(10, 10))
        plot = sjy.plot(column='d_fault')  # 直接使用 plt 的当前活动图形
        plt.title("D_Fault")
        plt.colorbar(plot.collections[0], label="D_Fault Feature Value")
        plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=4))  # 使用 plt.gca() 获取当前轴

        # 使用 Streamlit 显示图形
        st.pyplot(plt)

with placeholder5.container():
    f17,f18,f19,f20 = st.columns(4)

    with f17:
        plt.figure(figsize=(10, 10))
        plot = sjy.plot(column='d_river')  # 直接使用 plt 的当前活动图形
        plt.title("D_River")
        plt.colorbar(plot.collections[0], label="D_River Feature Value")
        plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=4))  # 使用 plt.gca() 获取当前轴

        # 使用 Streamlit 显示图形
        st.pyplot(plt)
    with f18:
        plt.figure(figsize=(10, 10))
        plot = sjy.plot(column='d_road')  # 直接使用 plt 的当前活动图形
        plt.title("D_Road")
        plt.colorbar(plot.collections[0], label="D_Road Feature Value")
        plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=4))  # 使用 plt.gca() 获取当前轴

        # 使用 Streamlit 显示图形
        st.pyplot(plt)
    with f19:
        plt.figure(figsize=(10, 10))
        plot = sjy.plot(column='ppv')  # 直接使用 plt 的当前活动图形
        plt.title("PPV")
        plt.colorbar(plot.collections[0], label="PPV Feature Value")
        plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=4))  # 使用 plt.gca() 获取当前轴

        # 使用 Streamlit 显示图形
        st.pyplot(plt)
    with f20:
        plt.figure(figsize=(10, 10))
        plot = sjy.plot(column='velocity')  # 直接使用 plt 的当前活动图形
        plt.title("Velocity")
        plt.colorbar(plot.collections[0], label="Velocity Feature Value")
        plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=4))  # 使用 plt.gca() 获取当前轴

        # 使用 Streamlit 显示图形
        st.pyplot(plt)

with placeholder6.container():
    f21,f22,f23,f24 = st.columns(4)

    with f21:
        plt.figure(figsize=(10, 10))
        plot = sjy.plot(column='lithology')  # 直接使用 plt 的当前活动图形
        plt.title("Lithology")
        plt.colorbar(plot.collections[0], label="Lithology Feature Value")
        plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=4))  # 使用 plt.gca() 获取当前轴

        # 使用 Streamlit 显示图形
        st.pyplot(plt)
    with f22:
        plt.figure(figsize=(10, 10))
        plot = sjy.plot(column='rock texture')  # 直接使用 plt 的当前活动图形
        plt.title("Rock Texture")
        plt.colorbar(plot.collections[0], label="Rock Texture Feature Value")
        plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=4))  # 使用 plt.gca() 获取当前轴

        # 使用 Streamlit 显示图形
        st.pyplot(plt)
    with f23:
        plt.figure(figsize=(10, 10))
        plot = sjy.plot(column='rock structure')  # 直接使用 plt 的当前活动图形
        plt.title("Rock Structure")
        plt.colorbar(plot.collections[0], label="Rock Structure Feature Value")
        plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=4))  # 使用 plt.gca() 获取当前轴

        # 使用 Streamlit 显示图形
        st.pyplot(plt)
    with f24:
        plt.figure(figsize=(10, 10))
        plot = sjy.plot(column='rfactor')  # 直接使用 plt 的当前活动图形
        plt.title("rfactor")
        plt.colorbar(plot.collections[0], label="rfactor Feature Value")
        plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=4))  # 使用 plt.gca() 获取当前轴

        # 使用 Streamlit 显示图形
        st.pyplot(plt)

st.markdown('**Lithology**: 0-中粗含砾石英砂岩,1-中风化黑云变粒岩,2-变质辉长岩,3-底砾岩,4-强风化黑云变粒岩,5-微风化黑云变粒岩,6-混合化黑云变粒岩,7-燧石白云岩,8-白云岩,\
    9-石英砂岩,10-石英砂岩与白云岩互层,11-覆盖土砂层  ')
st.markdown('**Rock Texture**: 0-中粒状变晶结构,1-中至粗粒结构,2-交代重结晶微晶结构,3-单粒结构,4-砂状结构,5-砾状结构,6-碎屑微晶结构,7-碎屑结构,8-粒状变晶结构,\
    9-细粒鳞片、粒状变晶结构,10-隐晶_自形粒状结构 ')
st.markdown('**Rock Structure**: 0-块状或弱片麻状构造,1-块状构造,2-砾状构造,3-裂隙状构造 ')

# 使用三引号来定义多行字符串，使用 Markdown 语法并在行尾添加两个空格来实现换行
multi_line_text = '''
特征基因中英名称对照：  
Height - 坡高  
Slope - 坡度  
Profile Curvature - 剖面曲率  
Lithology - 岩性  
Rock Texture - 岩石结构  
Rock Structure - 岩石构造  
D_Fault - 距断层距离  
D_River - 距河流距离  
PPV - 爆破最大质点震动速度  
D_Road - 距道路距离  
Velocity - 地表位移变化速率  
rfactor - 降雨侵蚀力  
  
'''
st.sidebar.markdown(multi_line_text)


st.title('特征统计分布')
''
''
# if st.button('View some random data'):
#     st.write(df.sample(5))

st.write(
    f'司家营边坡单元总数: {len(sjy)}. 1️⃣ 代表滑坡单元，0️⃣ 代表非滑坡单元 '
)
unbalancedf = pd.DataFrame(sjy.slide.value_counts())
st.write(unbalancedf)

# 需要一个count plot
placeholder = st.empty()
placeholder2 = st.empty()
placeholder3 = st.empty()



with placeholder.container():
    f1,f2,f3,f4 = st.columns(4)

    with f1:
        a11 = sjy[sjy['slide'] == 1]['height']
        a10 = sjy[sjy['slide'] == 0]['height']
        hist_data = [a10, a11]
        # group_labels = ['Real', 'Fake']
        fig = ff.create_distplot(hist_data, group_labels=['非滑坡', '滑坡'])
        fig.update_layout(title_text='坡高')
        st.plotly_chart(fig, use_container_width=True)
    with f2:
        a21 = sjy[sjy['slide'] == 1]['slope']
        a20 = sjy[sjy['slide'] == 0]['slope']
        hist_data = [a20, a21]
        #group_labels = ['Real', 'Fake']
        fig = ff.create_distplot(hist_data,group_labels = ['非滑坡', '滑坡'])
        fig.update_layout(title_text='坡度')
        st.plotly_chart(fig, use_container_width=True)
    with f3:
        a31 = sjy[sjy['slide'] == 1]['profile curvature']
        a30 = sjy[sjy['slide'] == 0]['profile curvature']
        hist_data = [a30, a31]
        # group_labels = []
        fig = ff.create_distplot(hist_data, group_labels=['非滑坡', '滑坡'])
        fig.update_layout(title_text='剖面曲率')
        st.plotly_chart(fig, use_container_width=True)
    with f4:
        a41 = sjy[sjy['slide'] == 1]['d_fault']
        a40 = sjy[sjy['slide'] == 0]['d_fault']
        hist_data = [a40, a41]
        # group_labels = []
        fig = ff.create_distplot(hist_data, group_labels=['非滑坡', '滑坡'])
        fig.update_layout(title_text='距断层距离')
        st.plotly_chart(fig, use_container_width=True)

with placeholder2.container():
    f5,f6,f7,f8 = st.columns(4)

    with f5:
        a51 = sjy[sjy['slide'] == 1]['d_river']
        a50 = sjy[sjy['slide'] == 0]['d_river']
        hist_data = [a50, a51]
        # group_labels = ['Real', 'Fake']
        fig = ff.create_distplot(hist_data, group_labels=['非滑坡', '滑坡'])
        fig.update_layout(title_text='距河流距离')
        st.plotly_chart(fig, use_container_width=True)
    with f6:
        a61 = sjy[sjy['slide'] == 1]['d_road']
        a60 = sjy[sjy['slide'] == 0]['d_road']
        hist_data = [a60, a61]
        #group_labels = ['Real', 'Fake']
        fig = ff.create_distplot(hist_data,group_labels = ['非滑坡', '滑坡'])
        fig.update_layout(title_text='距道路距离')
        st.plotly_chart(fig, use_container_width=True)
    with f7:
        a71 = sjy[sjy['slide'] == 1]['ppv']
        a70 = sjy[sjy['slide'] == 0]['ppv']
        hist_data = [a70, a71]
        # group_labels = []
        fig = ff.create_distplot(hist_data, group_labels=['非滑坡', '滑坡'])
        fig.update_layout(title_text='爆破最大质点震动速度')
        st.plotly_chart(fig, use_container_width=True)
    with f8:
        a81 = sjy[sjy['slide'] == 1]['velocity']
        a80 = sjy[sjy['slide'] == 0]['velocity']
        hist_data = [a80, a81]
        # group_labels = []
        fig = ff.create_distplot(hist_data, group_labels=['非滑坡', '滑坡'])
        fig.update_layout(title_text='地表位移变化速率')
        st.plotly_chart(fig, use_container_width=True)

df1 = sjy[['slide','lithology']].value_counts().reset_index(name='count')

df2 = sjy[['slide','rock texture']].value_counts().reset_index(name='count')

df3 = sjy[['slide','rock structure']].value_counts().reset_index(name='count')

df4 = sjy[['slide','rfactor']].value_counts().reset_index(name='count')

with placeholder3.container():
    f9,f10,f11,f12 = st.columns(4)

    with f9:
        #fig = plt.figure()
        fig = px.bar(df1, x='slide', y='count', color='lithology', color_continuous_scale=px.colors.qualitative.Plotly,  title=" 岩性: 共12种")
        st.write(fig)

    with f10:
        fig = px.bar(df2, x='slide', y='count', color="rock texture", title="岩石结构: 共11种")
        st.write(fig)

    with f11:
        fig = px.bar(df3, x='slide', y='count', color="rock structure", title="岩石构造: 共4种")
        st.write(fig)

    with f12:
        fig = px.bar(df4, x='slide', y='count', color="rfactor", title="降雨侵蚀力: 唯一值3733")
        st.write(fig)