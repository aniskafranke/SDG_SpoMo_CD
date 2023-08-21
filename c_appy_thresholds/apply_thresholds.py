import pandas as pd
import plotly.express as px

from config import RELEVANCE_THRESHOLD, CONFIDENCE_SCORE_THRESHOLD, EXTRACTABILITY_THRESHOLD


def main():
    snippet_path = "data/storage/classified_snippets.csv"
    filter_path = "data/storage/ex_lang.csv"

    df = pd.read_csv(snippet_path, sep=",",
                     engine='python', on_bad_lines='skip')

    df_filter = pd.read_csv(filter_path, sep=",",
                            engine='python', on_bad_lines='skip')


    df['companyname'] = df['companyname'].astype(str)
    companies = sorted(df['companyname'].unique())
    df = df.query('score >= @CONFIDENCE_SCORE_THRESHOLD').copy()

    filtered = pd.DataFrame(columns=['companyname', 'sdg'])

    for c in companies:

        df_c = df.query('companyname == @c')
        df_filter_c = df_filter.query('companyname == @c')

        if df_filter_c.query('isEng == False or extractability < @EXTRACTABILITY_THRESHOLD').empty:
            sdgs = sorted(df_c['sdg'].unique())
            # für alle enthaltenen SDGs der firma
            for sdg in sdgs:
                df_s = df_c.query('sdg == @sdg')

                # wenn SDG Anteil über Threshold, dann sichern
                if (len(df_s) / len(df_c) >= RELEVANCE_THRESHOLD):
                    new_row = {
                        'companyname': df_s['companyname'].values[0],
                        'sdg': df_s['sdg'].values[0]
                    }
                    filtered.loc[len(filtered.index)] = new_row

        else:
            print("Something is wrong with the files from " +
                  str(c) + ". Either the language is not English or the extractability is not given.")

    df.drop(df[df['sdg'] >= 17].index, inplace=True)
    fig = px.histogram(df, x="sdg", nbins=35, color="rep_type", title="SDG Distribution per report type (absolut)")
    fig.update_layout(
        xaxis=dict(
            tickmode='linear',
            tick0=1,
            dtick=1
        )
    )
    fig.show()

    fig1 = px.histogram(filtered, x="sdg", nbins=35, color="companyname",
                        title="Distribution of all relevant SDGs per company")
    fig1.update_layout(
        xaxis=dict(
            tickmode='linear',
            tick0=1,
            dtick=1
        )
    )
    fig1.show()
