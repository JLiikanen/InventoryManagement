import streamlit as st
import pandas as pd
import distributionFitting
from distributionFitting import randomDistributionData
import matplotlib.pyplot as plt
from fitter import Fitter
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
from streamlit_extras.metric_cards import style_metric_cards

# TO KEEP IN MIND:

# To unlock this, Streamlit apps have a unique data flow: any time something must be updated on the screen,
# Streamlit reruns your entire Python script from top to bottom.
# This can happen in two situations:
# Whenever you modify your app's source code.
# Whenever a user interacts with widgets in the app. For example, when dragging a slider, entering text in an input
# box, or clicking a button.

st.set_page_config(page_title="Capacity Management")

with open("styles.css") as f:
    st.markdown(f'<style> {f.read()} </style>', unsafe_allow_html=True)

# Global Variables

if 'dataOK' not in st.session_state:
    st.session_state['dataOk'] = False

data = None

st.write("<h1 style='text-align: center;'> Capacity Management With A Predictive  Model</h1>",
         unsafe_allow_html=True)
st.write('<p style="color:#7a7a7a;font-size:18px;text-align:center; margin-bottom: 4%;">10.4.2023 - Jacques '
         'Liikanen</p>',
         unsafe_allow_html=True)


def style_text(text):
    return f'<p style="color:#7a7a7a;font-size:18px;">{text}</p>'


st.write(style_text("This app takes a time-series dataset of your demand and fits a probability distribution to model "
                    "your market demand. Ideal for supply chain analytics &#x1F389;&#x1F389;"), unsafe_allow_html=True)

st.write('<span style="font-size:18px">**With the help of this app, you can:**</span>', unsafe_allow_html=True)

st.write("- Make better inventory management decisions \n"
         "- Better understand the nature of your demand \n"
         "- Determine the likelihood of your predictions \n"
         "- What chances do you want to take that your capacity won’t match demand?"
         " \n- Do you want to be 90% "
         "sure to meet demand? Check the capacity needed to meet demand with a 90% probability! \n"
         "- Improves performance analysis",
         unsafe_allow_html=True)

st.write('<span style="margin-top:5%">**```Got your own dataset?```** Input the data in a two-column format CSV file, '
         '(first column (Eg. time..), '
         'demand column).', unsafe_allow_html=True)
st.table(pd.DataFrame({"first colum": ["date1", "date2", "date3"], 'demand': [130, 140, 90]}), )

st.write("**```Want to test the app?```** Use our generator distributions to generate test data!")

st.write("The app uses the [Fitter Module](https://fitter.readthedocs.io/en/latest/references.html) to find the best-"
         "fitting distribution. The app tries to fit 14 different distributions to the data and picks the best one.")

# Large datasets are recommended to be splitted down

# Returns a subclass of BytesIO

delimiter = st.text_input(r"Enter the separator of your csv file.", placeholder="(Eg.   ' , '  /  ' ; ' )")
uploaded_data = st.file_uploader("Demand Data File", type=['csv'], help="Upload a CSV file of your data")

st.write("### Or")
st.write("Or test the app with randomly generated data from a probability distribution")
genMethod = st.selectbox("Select a Generation Method", options=
["Use My Own data",
 'Gamma',
 'Normal',
 'T',
 'Chisq',
 'Betaprime',
 'Weibull_min',
 'Weibull_max',
 'Burr12',
 'F',
 'Alpha',
 'Genlogistic',
 'Johnsonsb',
 'Johnsonsu',
 'Dgamma'
 ])

# HUOM DATASSA TÄRKEÄÄ ETTÄ HAVAINTOARVOJEN VÄLILLÄ ON AINA SAMA AIKA INTERVALLI

# Checking the data
if uploaded_data is not None:
    # Check that the data is good.
    # Read the CSV data into a Pandas DataFrame
    # Engine is a parser to automatically detect the delimiter
    try:
        data = pd.read_csv(uploaded_data, delimiter=delimiter, names=["time", "demand"])
        st.success("Upload received, Sir! => Press Start to Get your report!")
        st.session_state['dataOk'] = True
    except ValueError as e:
        st.warning("Did you forgot to specify a delimeter?")

green_light = st.button("Start Analysis")

if green_light:
    if genMethod == "Use My Own data":
        if st.session_state['dataOk']:
            pass
            # st.write("##### Starting analysis.... Preparing report...")
        else:
            st.warning("Upload data correctly.")
    else:
        # st.write(f"##### Starting analysis with data from a {genMethod} distribution.... Preparing report...")
        data = randomDistributionData(genMethod)
        st.session_state['dataOk'] = True

# Building the report!

if st.session_state['dataOk']:
    f = Fitter(data['demand'])
    f.distributions = ['gamma', 'norm', 't', 'chi2', 'betaprime', 'weibull_min',
                       'weibull_max', 'burr12', 'f', 'alpha', 'genlogistic', 'johnsonsb',
                       'johnsonsu', 'dgamma']
    with st.spinner("Starting analysis.... Preparing report..."):
        f.fit()

    st.write("## Real Demand Statistics")
    mean, std = st.columns(2)
    with mean:
        st.metric("Mean", round(np.mean((data['demand'])), 1), delta=None)
    with std:
        st.metric("Standard Deviation", round(np.std((data['demand'])), 1), delta=None)

    style_metric_cards()

    params = f.get_best()
    dist = next(iter(params))

    st.write(f"#### *The best distribution curve for your demand is the: {dist.capitalize()} distribution*")

    ranking = pd.DataFrame(f.df_errors['sumsquare_error'].sort_values())
    st.write("The sum of least squares was used to rank the models.")
    ranking.index = ranking.index.str.title()
    st.table(ranking.head(7))

    # ----------- dist hist plot ----------

    fig, ax = plt.subplots()
    ax.set_title(f"Real Demand and best-fitting PDF")
    ax.set_ylabel("Probability")
    ax.set_xlabel("Demand Units")

    ax.hist(data["demand"], edgecolor="black", density=True)

    f.plot_pdf(names=dist)

    st.pyplot(fig, clear_figure=True)

    # ------ INVENTORY LEVELS ----------
    st.write("### Key Capacity Levels")

    st.write("Use the plot's hover to discover the inventory level you need to be **89%** sure that you will "
             "meet your customers' needs!")
    # ------- CDF PLOT ---------

    cdf_y2 = [distributionFitting.distcdfvalue(x, params, dist.capitalize()) for x in f.x]

    # Add the histogram trace
    hist_trace = go.Histogram(
        x=data['demand'],
        # nbinsx=20,
        opacity=0.5,
        histnorm='probability density',
        name='Observed Demand',
        hoverinfo='skip',
        marker=dict(color="#1f77b4")
    )

    cdf_trace = go.Scatter(
        x=f.x,
        y=cdf_y2,
        name='CDF',
        hovertemplate='x: %{x:.2f}<br>y: %{y:.2f}',
        line=dict(color='#ff7f0e')
    )

    # Create a figure with two y-axes
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add the histogram trace to the first y-axis
    fig.add_trace(hist_trace)

    # Add the CDF trace to the second y-axis
    fig.add_trace(cdf_trace, secondary_y=True)

    # Set the x-axis label and range
    fig.update_xaxes(title_text='Demand', title_font=dict(size=18))

    # Set the y-axis label for the histogram
    fig.update_yaxes(title_text='Probability Density', secondary_y=False, title_font=dict(size=18))

    # Set the y-axis label for the CDF
    fig.update_yaxes(title_text='Cumulative Probability', secondary_y=True, title_font=dict(size=18))

    # Set the layout of the figure
    fig.update_layout(title='Histogram with CDF', legend=dict(x=0.02, y=0.95))
    # Create figure

    st.plotly_chart(fig, config={'displayModeBar': False, 'showAxisDragHandles': False}, use_container_width=True)

    # ------ SUMMARY PLOT ----------
    st.write("#### Other possible distributions")

    fig = plt.figure()
    table = f.summary()
    ax = fig.get_axes()[0]
    ax.set_title('Plotted Distributions')
    ax.set_xlabel('Demand Units')
    ax.set_ylabel('Probability')

    st.pyplot(fig, clear_figure=True)

    st.write("---")

    st.info('**Data Analyst: Jacques Liikanen**', icon="🕵")