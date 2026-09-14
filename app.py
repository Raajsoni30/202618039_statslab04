import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import statsmodels.api as sm

from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu, chi2_contingency, jarque_bera
from statsmodels.stats.outliers_influence import variance_inflation_factor

st.set_page_config(
    page_title="Medical Insurance Dashboard",
    layout="wide"
)

@st.cache_data
def load_data():
    return pd.read_csv("insurance(1).csv")

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
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()
    return model, X, y

df = load_data()
model, X, y = build_model(df)

st.title("Applied Statistical Modeling Dashboard")
st.caption("Medical Insurance Costs Dataset")

tab1, tab2, tab3 = st.tabs([
    "📊 Data Exploration",
    "🧪 Hypothesis Testing Lab",
    "📈 Live Prediction & Diagnostics"
])

with tab1:
    st.header("Data Exploration")

    st.sidebar.header("Filters")

    age_range = st.sidebar.slider(
        "Age Range",
        int(df.age.min()),
        int(df.age.max()),
        (int(df.age.min()), int(df.age.max()))
    )

    selected_regions = st.sidebar.multiselect(
        "Region",
        sorted(df.region.unique()),
        default=sorted(df.region.unique())
    )

    selected_smoker = st.sidebar.multiselect(
        "Smoking Status",
        sorted(df.smoker.unique()),
        default=sorted(df.smoker.unique())
    )

    filtered = df[
        df.age.between(age_range[0], age_range[1])
        & df.region.isin(selected_regions)
        & df.smoker.isin(selected_smoker)
    ]

    st.subheader("Dataset")
    st.dataframe(filtered, use_container_width=True)

    st.subheader("Summary Statistics")
    st.dataframe(filtered.describe(include="all").T, use_container_width=True)

    c1, c2 = st.columns(2)

    with c1:
        fig = px.histogram(
            filtered,
            x="charges",
            nbins=30,
            title="Distribution of Medical Charges"
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.scatter(
            filtered,
            x="bmi",
            y="charges",
            color="smoker",
            title="BMI vs Charges"
        )
        st.plotly_chart(fig, use_container_width=True)

    corr = filtered.select_dtypes(include=np.number).corr()

    fig = px.imshow(
        corr,
        text_auto=".2f",
        title="Correlation Matrix",
        aspect="auto"
    )
    st.plotly_chart(fig, use_container_width=True)


with tab2:
    st.header("Hypothesis Testing Lab")
    alpha = 0.05

    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()

    st.subheader("Compare Two Groups")

    factor = st.selectbox("Categorical Factor", categorical_cols)
    metric = st.selectbox("Numerical Metric", numerical_cols)

    groups = sorted(df[factor].dropna().unique())

    group1 = st.selectbox("Group 1", groups)
    group2 = st.selectbox(
        "Group 2",
        [g for g in groups if g != group1]
    )

    g1 = df.loc[df[factor] == group1, metric].dropna()
    g2 = df.loc[df[factor] == group2, metric].dropna()

    if len(g1) >= 3 and len(g2) >= 3:
        sh1 = shapiro(g1)
        sh2 = shapiro(g2)
        lev = levene(g1, g2)

        if sh1.pvalue > alpha and sh2.pvalue > alpha:
            result = ttest_ind(
                g1,
                g2,
                equal_var=(lev.pvalue > alpha)
            )
            test_name = "Two-Sample t-Test"
        else:
            result = mannwhitneyu(
                g1,
                g2,
                alternative="two-sided"
            )
            test_name = "Mann-Whitney U Test"

        st.write("Test Used:", test_name)
        st.metric("Test Statistic", f"{result.statistic:.4f}")
        st.metric("p-value", f"{result.pvalue:.6g}")

        if result.pvalue < alpha:
            st.success("Reject H₀ at α = 0.05.")
        else:
            st.info("Fail to Reject H₀ at α = 0.05.")

    st.divider()

    st.subheader("Chi-Square Test")

    cat1 = st.selectbox(
        "First Categorical Variable",
        categorical_cols,
        key="cat1"
    )

    cat2 = st.selectbox(
        "Second Categorical Variable",
        [c for c in categorical_cols if c != cat1],
        key="cat2"
    )

    table = pd.crosstab(df[cat1], df[cat2])
    chi2, p, dof, expected = chi2_contingency(table)

    st.dataframe(table)
    st.metric("Chi-Square Statistic", f"{chi2:.4f}")
    st.metric("p-value", f"{p:.6g}")

    if p < alpha:
        st.success("Reject H₀: variables are associated.")
    else:
        st.info("Fail to Reject H₀: insufficient evidence of association.")


with tab3:
    st.header("Live Prediction & Diagnostics")

    age = st.slider(
        "Age",
        int(df.age.min()),
        int(df.age.max()),
        int(df.age.median())
    )

    bmi = st.number_input(
        "BMI",
        min_value=float(df.bmi.min()),
        max_value=float(df.bmi.max()),
        value=float(df.bmi.median())
    )

    children = st.slider(
        "Children",
        int(df.children.min()),
        int(df.children.max()),
        int(df.children.median())
    )

    sex = st.selectbox("Sex", sorted(df.sex.unique()))
    smoker = st.selectbox("Smoking Status", sorted(df.smoker.unique()))
    region = st.selectbox("Region", sorted(df.region.unique()))

    if st.button("Predict Medical Charges"):
        user_data = pd.DataFrame([{
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "children": children,
            "smoker": smoker,
            "region": region
        }])

        encoded = pd.get_dummies(
            user_data,
            columns=["sex", "smoker", "region"],
            drop_first=True,
            dtype=int
        )

        encoded = encoded.reindex(
            columns=[c for c in X.columns if c != "const"],
            fill_value=0
        )

        encoded = sm.add_constant(encoded, has_constant="add")
        encoded = encoded.reindex(columns=X.columns, fill_value=0)

        pred = model.get_prediction(encoded).summary_frame(alpha=0.05).iloc[0]

        st.success(f"Predicted Charges: ${pred['mean']:,.2f}")

        c1, c2 = st.columns(2)

        c1.write("95% Confidence Interval")
        c1.write(
            f"${pred['mean_ci_lower']:,.2f} to "
            f"${pred['mean_ci_upper']:,.2f}"
        )

        c2.write("95% Prediction Interval")
        c2.write(
            f"${pred['obs_ci_lower']:,.2f} to "
            f"${pred['obs_ci_upper']:,.2f}"
        )

    st.divider()
    st.subheader("Residual Diagnostics")

    residuals = model.resid
    fitted = model.fittedvalues

    fig = px.scatter(
        x=fitted,
        y=residuals,
        labels={"x": "Fitted Values", "y": "Residuals"},
        title="Residuals vs Fitted Values"
    )
    fig.add_hline(y=0)
    st.plotly_chart(fig, use_container_width=True)

    qq = sm.qqplot(residuals, line="45", fit=True)
    st.pyplot(qq)

    jb = jarque_bera(residuals)
    st.write(f"Jarque-Bera Statistic: {jb.statistic:.4f}")
    st.write(f"Jarque-Bera p-value: {jb.pvalue:.6g}")

    X_vif = X.drop(columns=["const"])

    vif = pd.DataFrame({
        "Feature": X_vif.columns,
        "VIF": [
            variance_inflation_factor(X_vif.values, i)
            for i in range(X_vif.shape[1])
        ]
    })

    st.subheader("Multicollinearity (VIF)")
    st.dataframe(
        vif.sort_values("VIF", ascending=False),
        use_container_width=True
    )
