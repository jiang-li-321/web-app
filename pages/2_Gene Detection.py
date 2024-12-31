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
st.markdown("<h1 style='text-align: center; color: black;'>矿区边坡致滑特征基因诊断</h1>", unsafe_allow_html=True)


# 数据读取，建立变量
sjy = gpd.read_file(r"SJY_z_slide_v.geojson")
X_vars = ['height','slope','profile curvature','lithology','rock texture','rock structure','d_fault','d_river','ppv','d_road','velocity','rfactor']
x = sjy[X_vars]
x_copy = sjy[X_vars].copy()
y = sjy['slide']

# 类别数据编码
    # 创建一个字典来存储每个列的LabelEncoder实例和映射
encoders = {}
mappings = {}
    # 列表中的列需要编码
columns_to_encode = ['lithology', 'rock texture', 'rock structure']

for col in columns_to_encode:
    le = LabelEncoder()
    x_copy.loc[:, col] = le.fit_transform(x_copy[col])
    x_copy[col] = x_copy[col].astype('int64')
    encoders[col] = le
    mappings[col] = dict(zip(le.classes_, le.transform(le.classes_)))

# 多元耦合
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
poly_features = poly.fit_transform(x_copy)
feature_names = poly.get_feature_names_out()
x_copy_poly = pd.DataFrame(poly_features, columns=feature_names)

# 加载模型
loaded_model = joblib.load('SJY_slide.pkl')
train_x3, test_x3, train_y3, test_y3 = train_test_split(x_copy_poly, y, test_size=0.2, random_state=42)


# compute SHAP values
explainer = shap.TreeExplainer(loaded_model)
shap_values = explainer.shap_values(x_copy_poly)
shap_values2 = explainer(x_copy_poly)

#SHAP值空间分布
    # 选择正类的 SHAP 值
positive_class_shap_values = shap_values2[:, :, 1]
    # 将 SHAP 值添加到 GeoDataFrame
for i, var in enumerate(x_copy_poly):
    sjy[var + '_shap'] = positive_class_shap_values.values[:, i]

# 提取标签1的SHAP值
shap_values_label_1 = shap_values2.values[:, :, 1]  # 选择所有样本和特征，但只对标签1
# 创建一个新的SHAP Explanation 对象，只包含标签1的数据
explanation_label_1 = shap.Explanation(
    values=shap_values_label_1,
    base_values=shap_values2.base_values[:, 1],  # 选择标签1的基准值
    data=shap_values2.data,
    feature_names=shap_values2.feature_names
)

#side-bar
def user_input_features():
    st.sidebar.header('边坡致滑特征基因诊断')
    st.sidebar.write('全局诊断 ⬇️')

    select_var1 = st.sidebar.selectbox('诊断特征基因', x_copy_poly.columns)
    select_var2 = st.sidebar.selectbox('耦合特征基因', x_copy_poly.columns)

    # 使用三引号来定义多行字符串，使用 Markdown 语法并在行尾添加两个空格来实现换行
    multi_line_text = '''
    特征基因中英名称对照：  
    height - 坡高  
    slope - 坡度  
    profile curvature - 剖面曲率  
    lithology - 岩性  
    rock texture - 岩石结构  
    rock structure - 岩石构造  
    d_fault - 距断层距离  
    d_river - 距河流距离  
    ppv - 爆破最大质点震动速度  
    d_road - 距道路距离  
    velocity - 地表位移变化速率  
    rfactor - 降雨侵蚀力  
    特征^2 - 特征自耦合  
    特征1 特征2 - 特征互耦合  
    '''
    st.sidebar.markdown(multi_line_text)

    st.sidebar.write('个体诊断 ⬇️')
    sample_index = st.sidebar.number_input('样本ID: 0~1402', min_value=0, max_value=1402, step=1)
    #sample_1_index = st.sidebar.selectbox('滑坡样本ID', x_copy_poly.columns)

    output = [select_var1, select_var2, sample_index]
    return output

# 调用函数并解包返回的值
select_var1, select_var2, sample_index = user_input_features()

# 找到当前变量在x_copy_poly中的索引
# var_index = x_copy_poly.index(var)
var_index = x_copy_poly.columns.get_loc(select_var1)

# 获取对应的数据
x_spline = shap_values2[:, var_index, 1].data  # 特征值
y_spline = shap_values2[:, var_index, 1].values  # SHAP值



st.title('特征重要性')
''
''

placeholder = st.empty()

with placeholder.container():
    f1, f2 = st.columns(2)

    with f1:
        st.markdown('''
            **⬇️说明**：SHAP分析的汇总图，展示每个特征的SHAP值。  
            - **X轴**：SHAP值的大小，SHAP>0表示预测更趋向于滑坡。   
            - **Y轴**：左侧为按重要度排序的特征名称，右侧蓝-红色带表示特征值由低到高。  
            - 图中每个蜂群点代表一个样本。
        ''')

        shap.summary_plot(shap_values[1], x_copy_poly)
        st.pyplot(plt.gcf())
        '***'

    with f2:

        st.markdown('**⬇️说明**：该特征重要度排序是依据LightGBM模型训练而得。')
        ''
        ''
        ''
        ''
        fig, ax = plt.subplots(figsize=(10, 14.9))
        plot_importance(loaded_model, height=.5, ax=ax, max_num_features=20)  # max_num_features=10
        plt.title("Feature Importances")
        st.pyplot(fig)
        '***'

st.title('特征基因全局诊断图')
''
''

#if st.button('Global Diagnose'):

placeholder2 = st.empty()
placeholder3 = st.empty()

with placeholder2.container():
    f3, f4, f5 = st.columns(3)

    with f3:
        fig, ax = plt.subplots(figsize=(10, 10))

        # 获取数据范围
        column_name = select_var1 + '_shap'
        data = sjy[column_name]
        vmax = np.max(np.abs(data))  # 找到最大绝对值
        vmin = -vmax
        # 在ax上绘制地理数据
        plot = sjy.plot(column=column_name, legend=False, cmap='RdBu_r', vmin=vmin, vmax=vmax, ax=ax)
        ax.set_title(select_var1)
        # 添加颜色条
        cbar = fig.colorbar(plot.collections[0], ax=ax, label="SHAP Value for " + select_var1)
        cbar.set_label("SHAP Value for " + select_var1)
        #plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=4))  # 使用 plt.gca() 获取当前轴

        # 使用 Streamlit 显示图形
        st.pyplot(fig)
        st.markdown('''
            **说明：诊断特征基因SHAP值空间分布图，**  
            - 图例色带中白色为SHAP=0，红色带SHAP>0，蓝色带SHAP<0。  
            - 红色带表示该诊断特征值不利于边坡的稳定，属于该边坡致滑基因。红色越深，表示该基因致滑度越高。     
        ''')
        st.markdown('**建议：深红色边坡的诊断特征基因需要着重注意**')

        '***'

    with f4:
        shap.plots.scatter(shap_values2[:, select_var1, 1], color=shap_values2[:,select_var2,1])
        st.pyplot(plt.gcf())
        ''
        ''
        ''
        ''
        ''
        ''
        ''
        ''
        ''
        ''
        st.markdown('''
            **说明：两个变量交互的散点图。**
            展示某两个特征交互的SHAP值。
            - **X轴**：为诊断特征基因的值，底部阴影为该特征样本量的统计分布。   
            - **Y轴**：右侧为耦合特征基因的值，左侧为诊断特征基因的SHAP值。  
            - 以Y轴SHAP=0为界限，SHAP值大于0，说明该特征基因的值不利于边坡稳定。
        ''')

        '***'

    with f5:
        fig, ax = plt.subplots(figsize=(10, 10))
        column_name = select_var1 + '_shap'
        data = sjy[column_name]
        # 绘制直方图，正值和负值使用不同颜色
        ax.hist(data[data > 0], bins=20, color='#6E7074', alpha=0.7, label='SHAP > 0')
        ax.hist(data[data < 0], bins=20, color='#C0B2B5', alpha=0.7, label='SHAP < 0')

        # 在 SHAP=0 的位置画一条红色虚线
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2)

        # 计算 SHAP>0 的样本占比并显示
        positive_ratio = np.sum(data > 0) / len(data) * 100
        ax.text(0.05, 0.95, f'Positive SHAP: {positive_ratio:.2f}%', transform=ax.transAxes, verticalalignment='top', fontsize=15)
        ax.set_title(f"Histogram of {column_name}")
        ax.legend(loc='upper right', fontsize=15)
        st.pyplot(fig)
        ''
        ''
        ''
        ''
        st.markdown('''
            **说明：诊断特征基因的SHAP值频数分布图。**    
            - **X轴**：为诊断特征基因的SHAP值。   
            - **Y轴**：为对应诊断特征基因SHAP值的样本量统计值。  
            - **Positive SHAP**：为SHAP>0的样本量的统计占比，即该诊断特征值在全矿区边坡中有多少超过危险阈值。
        ''')

        '***'

with placeholder3.container():
    f6, f7 = st.columns(2)

    with f6:
        n_knots = st.slider("n_knots：结点的数量", 2, 10, 3, 1)
        degree = st.slider("degree：多项式最高次数", 1, 10, 3, 1)
        model_spline = make_pipeline(SplineTransformer(n_knots=n_knots, degree=degree), Ridge(alpha=1e-3))
        model_spline.fit(x_spline.reshape(-1, 1), y_spline)
        y_plot = model_spline.predict(x_spline.reshape(-1, 1))
        plot_data = pd.DataFrame({
            select_var1: x_spline,
            'shap_' + select_var1: y_spline,
            'spline': y_plot
        })
        plot_data = plot_data.sort_values(by=select_var1)
        annotation_number = st.number_input("左竖线位置", min_value=float(min(plot_data[select_var1])), max_value=float(max(plot_data[select_var1])),
                                            value=float(np.median(plot_data[select_var1])), step=1.0)
        annotation_number2 = st.number_input("右竖线位置", min_value=float(min(plot_data[select_var1])), max_value=float(max(plot_data[select_var1])),
                                             value=5 + float(np.median(plot_data[select_var1])), step=1.0)
        ''
        st.markdown('**说明**：使用限制性立方样条曲线对诊断特征基因SHAP值散点图进行拟合。')
        st.markdown('用于确定关键点对应的特征值，即致滑基因阈值，包括SHAP=0的点对应的特征值，曲线拐点特征值。')

        '***'
    with f7:

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=plot_data[select_var1], y=plot_data['shap_' + select_var1], mode='markers',
                                  name='shap_' + select_var1))
        fig2.add_trace(
            go.Scatter(x=plot_data[select_var1], y=plot_data['spline'], mode='lines', name='spline',
                       line=dict(color='green')))
        fig2.add_hline(y=0, line_dash="dash", line_color="red")
        fig2.add_vline(x=annotation_number, line_dash='dash', annotation_text='{}'.format(annotation_number))
        fig2.add_vline(x=annotation_number2, line_dash='dash', annotation_text='{}'.format(annotation_number2))
        st.plotly_chart(fig2)
        ''
        '***'



st.title('特征基因个体诊断图')
''
''
if st.button('Individual Diagnose'):

    placeholder4 = st.empty()
    placeholder5 = st.empty()
    placeholder6 = st.empty()

    with placeholder4.container():
        f8, f9 = st.columns(2)
        with f8:

            st.markdown('**⬇️说明：该图为矿区所有边坡样本个体ID**')
            # 加载本地TIFF图片
            # 注意：Streamlit 原生不支持直接显示TIFF格式，需要先转换为支持的格式
            img = Image.open(r"xiepo_id0.tiff")
            st.image(img)#caption="This is a TIFF image"
            '***'

        with f9:

            st.markdown('**⬇️说明：该图为矿区现有及预测危险边坡位置及ID图**')
            ''
            ''
            ''
            ''
            ''
            ''
            ''
            ''
            ''
            ''
            img = Image.open(r"SJY_tietu.tif")
            st.image(img)
            ''
            ''
            ''
            ''
            ''
            ''
            ''
            ''
            ''
            '***'

    with placeholder5.container():
        f10, f11 = st.columns(2)

        with f10:
            fig, ax = plt.subplots()
            shap.plots.waterfall(explanation_label_1[sample_index], max_display=20, show=False)  # max_display 控制显示的特征数量
            st.pyplot(fig)
            st.markdown('''
                **说明：个体样本特征基因瀑布图。**  
                从基线值E[f(x)]开始，逐个特征地累加每个特征的SHAP值，得出实际预测值f(x)。  
                - **E[f(x)]**：是模型输出的平均值或期望值，即基线预测值。 
                - **f(x)**：是模型对特定输入样本的实际预测值，是考虑了所有输入特征后模型计算出的输出值。
                - **SHAP 值**：解释了每个特征对于从基线预测 E[f(x)] 到实际预测 f(x) 的贡献是增加还是减少。
            ''')
            st.markdown('**建议：图中红色特征基因为该边坡样本的致滑特征基因，是预防、治理的要点**')

            '***'

        with f11:
            fig, ax = plt.subplots()
            shap.decision_plot(explainer.expected_value[1], shap_values[1][sample_index, :], x_copy_poly.iloc[sample_index, :], show=False)
            #shap.plots.waterfall(explanation_label_1[sample_index], show=False)  # max_display 控制显示的特征数量
            st.pyplot(fig)
            ''
            ''
            ''
            ''
            ''
            ''
            st.markdown('''
                **说明：个体样本特征基因决策图。**  
                即特征值是如何累积影响模型的最终决策。    
                决策图提供了从基线到最终预测的连续视觉路径，有助于揭示模型在整个输入空间中的行为模式。   
                '决策图中决策路径线旁边的数字为各特征基因的值'
            ''')

            '***'

    with placeholder6.container():
        f12 = st.columns(1)
        with f12[0]:

            shap.force_plot(explainer.expected_value[1], shap_values[1][sample_index, :], x_copy_poly.iloc[sample_index, :], show=False, matplotlib=True)
            st.pyplot(plt.gcf())
            st.markdown('''
                **说明：个体样本特征基因链驱动力图。**
                展示了每个特征基因对模型预测的贡献是正面还是负面，以及这些贡献的相对大小。  
                - **红色特征基因箭头**：表示正的 SHAP 值，意味着该特征基因使预测值增加。 
                - **蓝色特征基因箭头**：表示负的 SHAP 值，意味着该特征基因使预测值减少。
                - **长度**：每个特征基因箭头的长度表示该特征基因的影响力大小。
            ''')

            '***'



