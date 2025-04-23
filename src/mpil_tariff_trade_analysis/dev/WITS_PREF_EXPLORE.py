import marimo

__generated_with = "0.12.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import altair as alt
    import pandas as pd
    return alt, mo, pd, pl


@app.cell
def _(pl):
    # Load the wits pref data. Do we have issues in it?

    lf = pl.scan_parquet("data/intermediate/WITS_AVEPref.parquet")

    lf.describe()
    return (lf,)


@app.cell
def _(lf):
    df = lf.collect().to_pandas()
    return (df,)


@app.cell
def _(df, pd):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import math

    def profile_dataframe_plotly(dataframe: pd.DataFrame) -> go.Figure:
        """
        Generates a Plotly figure with subplots (histograms for numeric,
        bar charts for categorical) to profile a pandas DataFrame.
        Excludes columns with only one unique value.

        Args:
            dataframe: The pandas DataFrame to profile. Should be a pandas DataFrame.

        Returns:
            A Plotly Figure object.
        """
        # --- Check input type ---
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(f"Input must be a pandas DataFrame, but got {type(dataframe)}")

        traces = []
        subplot_titles = []
        valid_columns_count = 0

        # --- Identify numeric columns and create histograms ---
        numeric_cols = dataframe.select_dtypes(include=['number']).columns.tolist()
        for col in numeric_cols:
            if dataframe[col].nunique() > 1:
                traces.append(go.Histogram(x=dataframe[col], name=col.replace('_', ' ').title()))
                subplot_titles.append(f'Distribution of {col.replace("_", " ").title()}')
                valid_columns_count += 1

        # --- Identify categorical columns and create bar charts ---
        categorical_cols = dataframe.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in categorical_cols:
            if dataframe[col].nunique() > 1:
                counts = dataframe[col].value_counts()
                traces.append(go.Bar(x=counts.index, y=counts.values, name=col.replace('_', ' ').title()))
                subplot_titles.append(f'Frequency of {col.replace("_", " ").title()}')
                valid_columns_count += 1

        # --- Create subplots ---
        if valid_columns_count == 0:
            fig = go.Figure()
            fig.update_layout(
                title="Dataset Profile",
                xaxis_showgrid=False, xaxis_zeroline=False, xaxis_visible=False,
                yaxis_showgrid=False, yaxis_zeroline=False, yaxis_visible=False,
                annotations=[
                    go.layout.Annotation(
                        text="No suitable columns found for profiling.",
                        xref="paper", yref="paper", showarrow=False, font=dict(size=14)
                    )
                ]
            )
            return fig

        cols = 2
        rows = math.ceil(valid_columns_count / cols)
        fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles)

        trace_index = 0
        for r in range(1, rows + 1):
            for c in range(1, cols + 1):
                if trace_index < len(traces):
                    fig.add_trace(traces[trace_index], row=r, col=c)
                    trace_index += 1
                else:
                     fig.update_xaxes(visible=False, row=r, col=c)
                     fig.update_yaxes(visible=False, row=r, col=c)

        fig.update_layout(
            title_text='Dataset Profile',
            height=350 * rows, # Adjust height dynamically
            showlegend=False,
            margin=dict(t=100)
        )

        return fig


    profile_chart = profile_dataframe_plotly(df)
    return go, make_subplots, math, profile_chart, profile_dataframe_plotly


@app.cell
def _(profile_chart):
    profile_chart.write_html("wits_pref_charts.html")
    return


@app.cell
def _(df):
    df.sample(1000000)
    return


if __name__ == "__main__":
    app.run()
