import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.api as sm

from scipy.stats import (
    shapiro,
    levene,
    ttest_ind,
    mannwhitneyu,
    chi2_contingency,
    jarque_bera
)

from statsmodels.stats.outliers_influence import variance_inflation_factor


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Medical Insurance Dashboard",
    layout="wide"
)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():
    return pd.read_csv("insurance.csv")


df = load_data()


# =========================================================
# BUILD REGRESSION MODEL
# =========================================================

@st.cache_resource
def build_model(data):

    model_df = pd.get_dummies(
        data,
        columns=["sex", "smoker", "region"],
        drop_first=True,
        dtype=int
    )

    X = model_df.drop(columns=["charges"])
    y = model_df["charges"]

    # Add intercept
    X = sm.add_constant(X)

    # OLS Regression
    model = sm.OLS(y, X).fit()

    return model, X, y


model, X, y = build_model(df)


# =========================================================
# TITLE
# =========================================================

st.title("Applied Statistical Modeling & Interactive Web Dashboard")

st.write(
    "### Medical Insurance Costs Dataset"
)

st.write(
    "M.Sc. Data Science — Statistical Modeling with Python"
)


# =========================================================
# CREATE TABS
# =========================================================

tab1, tab2, tab3 = st.tabs([

    "📊 Data Exploration",

    "🧪 Hypothesis Testing Lab",

    "📈 Live Prediction & Diagnostics"

])


# =========================================================
# TAB 1 — DATA EXPLORATION
# =========================================================

with tab1:

    st.header("Data Exploration")

    # -------------------------
    # SIDEBAR FILTERS
    # -------------------------

    st.sidebar.header("Interactive Filters")

    age_range = st.sidebar.slider(

        "Select Age Range",

        int(df["age"].min()),

        int(df["age"].max()),

        (
            int(df["age"].min()),

            int(df["age"].max())
        )
    )


    selected_regions = st.sidebar.multiselect(

        "Select Region",

        options=sorted(df["region"].unique()),

        default=sorted(df["region"].unique())
    )


    selected_smoker = st.sidebar.multiselect(

        "Smoking Status",

        options=sorted(df["smoker"].unique()),

        default=sorted(df["smoker"].unique())
    )


    # -------------------------
    # FILTER DATA
    # -------------------------

    filtered_df = df[

        df["age"].between(
            age_range[0],
            age_range[1]
        )

        & df["region"].isin(selected_regions)

        & df["smoker"].isin(selected_smoker)

    ]


    # -------------------------
    # DATASET
    # -------------------------

    st.subheader("Filtered Dataset")

    st.dataframe(
        filtered_df,
        use_container_width=True
    )


    # -------------------------
    # SUMMARY
    # -------------------------

    st.subheader("Summary Statistics")

    st.dataframe(
        filtered_df.describe().T,
        use_container_width=True
    )


    # -------------------------
    # PLOTS
    # -------------------------

    col1, col2 = st.columns(2)


    with col1:

        fig = px.histogram(

            filtered_df,

            x="charges",

            nbins=30,

            title="Distribution of Medical Charges"

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    with col2:

        fig = px.scatter(

            filtered_df,

            x="bmi",

            y="charges",

            color="smoker",

            title="BMI vs Medical Charges"

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # -------------------------
    # CORRELATION MATRIX
    # -------------------------

    st.subheader("Correlation Matrix")

    numeric_df = filtered_df.select_dtypes(
        include=np.number
    )

    corr = numeric_df.corr()


    fig = px.imshow(

        corr,

        text_auto=".2f",

        title="Correlation Matrix",

        aspect="auto"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================================================
# TAB 2 — HYPOTHESIS TESTING
# =========================================================

with tab2:

    st.header("Hypothesis Testing Lab")

    alpha = 0.05


    # =====================================================
    # TWO GROUP TEST
    # =====================================================

    st.subheader("Compare Two Groups")


    categorical_cols = df.select_dtypes(
        include=["object"]
    ).columns.tolist()


    numerical_cols = df.select_dtypes(
        include=np.number
    ).columns.tolist()


    factor = st.selectbox(

        "Select Categorical Factor",

        categorical_cols

    )


    metric = st.selectbox(

        "Select Numerical Metric",

        numerical_cols

    )


    groups = sorted(
        df[factor].dropna().unique()
    )


    group1 = st.selectbox(

        "Group 1",

        groups

    )


    remaining_groups = [

        g for g in groups

        if g != group1

    ]


    group2 = st.selectbox(

        "Group 2",

        remaining_groups

    )


    g1 = df.loc[
        df[factor] == group1,
        metric
    ].dropna()


    g2 = df.loc[
        df[factor] == group2,
        metric
    ].dropna()


    # -------------------------
    # SHAPIRO TEST
    # -------------------------

    shapiro1 = shapiro(g1)

    shapiro2 = shapiro(g2)


    # -------------------------
    # LEVENE TEST
    # -------------------------

    levene_result = levene(
        g1,
        g2
    )


    # -------------------------
    # SELECT TEST
    # -------------------------

    if (

        shapiro1.pvalue > alpha

        and

        shapiro2.pvalue > alpha

    ):

        result = ttest_ind(

            g1,

            g2,

            equal_var=(
                levene_result.pvalue > alpha
            )

        )

        test_name = "Two-Sample t-Test"


    else:

        result = mannwhitneyu(

            g1,

            g2,

            alternative="two-sided"

        )

        test_name = "Mann-Whitney U Test"


    # -------------------------
    # RESULTS
    # -------------------------

    st.write("### Test Results")

    st.write(
        "**Test Used:**",
        test_name
    )

    st.write(
        "**Test Statistic:**",
        round(result.statistic, 4)
    )

    st.write(
        "**p-value:**",
        result.pvalue
    )


    if result.pvalue < alpha:

        st.success(

            "Reject H₀ at α = 0.05. "

            "There is a statistically significant difference."

        )

    else:

        st.info(

            "Fail to Reject H₀ at α = 0.05. "

            "There is insufficient evidence of a significant difference."

        )


    # =====================================================
    # CHI-SQUARE TEST
    # =====================================================

    st.divider()

    st.subheader(
        "Chi-Square Test of Association"
    )


    cat1 = st.selectbox(

        "Categorical Variable 1",

        categorical_cols,

        key="cat1"

    )


    cat2_options = [

        c for c in categorical_cols

        if c != cat1

    ]


    cat2 = st.selectbox(

        "Categorical Variable 2",

        cat2_options,

        key="cat2"

    )


    contingency_table = pd.crosstab(

        df[cat1],

        df[cat2]

    )


    chi2, p, dof, expected = chi2_contingency(
        contingency_table
    )


    st.dataframe(
        contingency_table
    )


    st.write(
        "**Chi-Square Statistic:**",
        round(chi2, 4)
    )


    st.write(
        "**p-value:**",
        p
    )


    if p < alpha:

        st.success(

            "Reject H₀: "

            "The variables are statistically associated."

        )

    else:

        st.info(

            "Fail to Reject H₀: "

            "There is insufficient evidence of association."

        )


# =========================================================
# TAB 3 — PREDICTION & DIAGNOSTICS
# =========================================================

with tab3:

    st.header(
        "Live Prediction & Diagnostics"
    )


    # =====================================================
    # USER INPUT
    # =====================================================

    st.subheader(
        "Predict Medical Charges"
    )


    age = st.slider(

        "Age",

        int(df["age"].min()),

        int(df["age"].max()),

        int(df["age"].median())

    )


    bmi = st.number_input(

        "BMI",

        min_value=float(df["bmi"].min()),

        max_value=float(df["bmi"].max()),

        value=float(df["bmi"].median())

    )


    children = st.slider(

        "Children",

        int(df["children"].min()),

        int(df["children"].max()),

        int(df["children"].median())

    )


    sex = st.selectbox(

        "Sex",

        sorted(df["sex"].unique())

    )


    smoker = st.selectbox(

        "Smoking Status",

        sorted(df["smoker"].unique())

    )


    region = st.selectbox(

        "Region",

        sorted(df["region"].unique())

    )


    # =====================================================
    # PREDICTION
    # =====================================================

    if st.button(
        "Predict Medical Charges"
    ):


        user_data = pd.DataFrame([{

            "age": age,

            "sex": sex,

            "bmi": bmi,

            "children": children,

            "smoker": smoker,

            "region": region

        }])


        # Convert categorical variables

        user_encoded = pd.get_dummies(

            user_data,

            columns=[
                "sex",
                "smoker",
                "region"
            ],

            drop_first=True,

            dtype=int

        )


        # Match training columns

        user_encoded = user_encoded.reindex(

            columns=[
                col for col in X.columns
                if col != "const"
            ],

            fill_value=0

        )


        # Add intercept

        user_encoded = sm.add_constant(

            user_encoded,

            has_constant="add"

        )


        user_encoded = user_encoded.reindex(

            columns=X.columns,

            fill_value=0

        )


        # Prediction

        prediction = model.get_prediction(
            user_encoded
        )


        prediction_df = prediction.summary_frame(
            alpha=0.05
        )


        pred = prediction_df.iloc[0]


        st.success(

            f"Predicted Medical Charges: "

            f"${pred['mean']:,.2f}"

        )


        col1, col2 = st.columns(2)


        with col1:

            st.write(
                "### 95% Confidence Interval"
            )

            st.write(

                f"${pred['mean_ci_lower']:,.2f}"

                f" to "

                f"${pred['mean_ci_upper']:,.2f}"

            )


        with col2:

            st.write(
                "### 95% Prediction Interval"
            )

            st.write(

                f"${pred['obs_ci_lower']:,.2f}"

                f" to "

                f"${pred['obs_ci_upper']:,.2f}"

            )


    # =====================================================
    # DIAGNOSTICS
    # =====================================================

    st.divider()

    st.subheader(
        "Regression Diagnostics"
    )


    fitted_values = model.fittedvalues

    residuals = model.resid


    # -------------------------
    # RESIDUAL PLOT
    # -------------------------

    fig = px.scatter(

        x=fitted_values,

        y=residuals,

        labels={

            "x": "Fitted Values",

            "y": "Residuals"

        },

        title="Residuals vs Fitted Values"

    )


    fig.add_hline(
        y=0
    )


    st.plotly_chart(

        fig,

        use_container_width=True

    )


    # -------------------------
    # Q-Q PLOT
    # -------------------------

    st.subheader(
        "Q-Q Plot of Residuals"
    )


    qq_fig = sm.qqplot(

        residuals,

        line="45",

        fit=True

    )


    st.pyplot(
        qq_fig
    )


    # -------------------------
    # JARQUE-BERA
    # -------------------------

    st.subheader(
        "Jarque-Bera Normality Test"
    )


    jb = jarque_bera(
        residuals
    )


    st.write(
        "Jarque-Bera Statistic:",
        round(jb.statistic, 4)
    )


    st.write(
        "p-value:",
        jb.pvalue
    )


    # =====================================================
    # VIF
    # =====================================================

    st.subheader(
        "Multicollinearity — VIF"
    )


    X_vif = X.drop(
        columns=["const"]
    )


    vif_data = pd.DataFrame({

        "Feature": X_vif.columns,

        "VIF": [

            variance_inflation_factor(

                X_vif.values,

                i

            )

            for i in range(
                X_vif.shape[1]
            )

        ]

    })


    st.dataframe(

        vif_data.sort_values(

            "VIF",

            ascending=False

        ),

        use_container_width=True

    )