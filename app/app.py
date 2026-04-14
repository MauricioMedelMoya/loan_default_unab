from dash import Dash, html

app = Dash(__name__)

app.layout = html.Div([
    html.H1("Loan Default Dashboard"),
    html.P("Proyecto en desarrollo")
])

if __name__ == "__main__":
    app.run(debug=True)
