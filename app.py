import os
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sch_env import SchEnv
from info_2020_scheduling.Mathis.train import *
from info_2020_scheduling.Mathis.agentsch import *
from info_2020_scheduling.Pierre.PPG_sans_entropy.manon_agent_ppg import *
from info_2020_scheduling.Pierre.PPG_sans_entropy.manon_exec_ppg import *
import pandas as pd
import json
import gym

with open("info_2020_scheduling/Mathis/PraindeMine.json", "r") as fichier:
    ENV = json.load(fichier)
    Name = list(ENV["products"].keys())

###CODE PRE LAYOUT
external_stylesheets = [
    {
        "href": "https://unpkg.com/purecss@1.0.1/build/pure-min.css",
        "rel": "stylesheet",
        "integrity": "sha384-",
        "crossorigin": "anonymous",
    },
    "https://unpkg.com/purecss@1.0.1/build/grids-responsive-min.css",
]
# meta_tags = [{"name": "viewport", "content": "width=device-width, initial-scale=0.5"}]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

options = [
    {"label": "PPO", "value": "PPO"},
    {"label": "Deep Q-Learning", "value": "DQL"},
]

sidebar = html.Div(
    [
        dcc.Store(id="data-store"),
        html.Div(
            children=[
                html.H1("Mines Production Scheduling", className="brand-title"),
                html.Nav(
                    html.Ul(
                        [
                            html.Li(
                                [
                                    html.H2("Choose Agent:", className="brand-tagline"),
                                    dcc.Dropdown(
                                        id="dropdown", options=options, value="DQL"
                                    ),
                                ]
                            ),
                            html.Li(
                                [
                                    html.H2("Stock Farine", className="brand-tagline"),
                                    dcc.Slider(
                                        id="slider1",
                                        min=0,
                                        max=20,
                                        step=1,
                                        value=10,
                                        marks={
                                            i: {
                                                "label": "%i" % i,
                                                "style": dict(color="#fff"),
                                            }
                                            for i in [0, 5, 10, 15, 20]
                                        },
                                    ),
                                ], className="nav-item"
                            ),
                            html.Li(
                                [
                                    html.H2("Stock Plastique", className="brand-tagline"),
                                    dcc.Slider(
                                        id="slider2",
                                        min=0,
                                        max=20,
                                        step=1,
                                        value=10,
                                        marks={
                                            i: {
                                                "label": "%i" % i,
                                                "style": dict(color="#fff"),
                                            }
                                            for i in [0, 5, 10, 15, 20]
                                        },
                                    ),
                                ], className="nav-item"
                            ),
                            html.Li(
                                [
                                    html.H2("Demande Pain de Mie", className="brand-tagline"),
                                    dcc.Slider(
                                        id="slider3",
                                        min=0,
                                        max=20,
                                        step=1,
                                        value=5,
                                        marks={
                                            i: {
                                                "label": "%i" % i,
                                                "style": dict(color="#fff"),
                                            }
                                            for i in [0, 5, 10, 15, 20]
                                        },
                                    ),
                                ], className="nav-item"
                            ),
                            html.Li(
                                [
                                    html.H2(
                                        "Demande Pain de Mie Sans Croute",
                                        className="brand-tagline",
                                    ),
                                    dcc.Slider(
                                        id="slider4",
                                        min=0,
                                        max=20,
                                        step=1,
                                        value=5,
                                        marks={
                                            i: {
                                                "label": "%i" % i,
                                                "style": dict(color="#fff"),
                                            }
                                            for i in [0, 5, 10, 15, 20]
                                        },
                                    ),
                                ], className="nav-item"
                            ),
                            html.Div([html.Button("RUN", id="my-button", n_clicks=0, className="pure-button"),], className="nav-item"),
                        ],
                        className="nav-list",
                    ),
                    className="nav",
                ),
            ],
            className="header",
        ),
    ],
    className="sidebar pure-u-1 pure-u-md-6-24",
)

output = html.Div(
    [
        html.Div(
            [
                html.Div(
                    [
                        html.H1("Problem Definition", className="content-subhead"),
                        # ===================== Post ===========================
                        html.Section(
                            [
                                html.Header([
                                     html.Img(className="post-avatar", src="/assets/jb_avatar.png", style={'height':'48', 'width':'48'}),
                                     html.Div("Client Need", className="post-title"),
                                     html.P([
                                         "By ", html.A("JB", className="post-author", href="#"),
                                         " - ", html.A("Client", className="post-category")
                                         ], className='post-meta'),
                                        ], className="post-header",
                                ),
                                html.Div([
                                    html.P('"Je dois planifier ma production de pain de mie. Il faut que le optimise mes couts de productions en fonction de la demande."'),
                                    html.Div([
                                        # Update 
                                        html.Img(src="./assets/Production_line_example.png", className='pure-img-responsive'),
                                        ], className='post-images pure-g')
                                    ], className='post-description')],
                            className="post",
                        ),
                        # ===================== Post ===========================
                        html.H1("Reinforcement Learning", className="content-subhead"),
                        # ===================== Post ===========================
                        html.Section(
                            [
                                html.Header([
                                     html.Img(className="post-avatar", src="/assets/manon_avatar.png", style={'height':'48', 'width':'48'}),
                                     html.Div("Reinforcement Learning", className="post-title"),
                                     html.P([
                                         "By ", html.A("Manon", className="post-author", href="#"),
                                         " - ", html.A("Mines", className="post-category post-category-pure")
                                         ], className='post-meta'),
                                        ], className="post-header",
                                ),
                                html.Div([
                                    html.Div([
                                        html.Img(src="/assets/RL.png", className='pure-img-responsive'),
                                        ], className='post-images pure-g')
                                    ], className='post-description')
                            ],
                            className="post",
                        ),
                        # ===================== Post ===========================
                        # ===================== Post ===========================
                        html.Section(
                            [
                                html.Header([
                                     html.Img(className="post-avatar", src="/assets/pierre_avatar.png", style={'height':'48', 'width':'48'}),
                                     html.Div("Policy Gradient", className="post-title"),
                                     html.P([
                                         "By ", html.A("Pierre", className="post-author", href="#"),
                                         " - ", html.A("Mines", className="post-category post-category-pure")
                                         ], className='post-meta'),
                                        ], className="post-header",
                                ),
                                html.Div([
                                    html.P(["Mettre Une description ici Pierre"]),
                                    html.Div([
                                        # Update 
                                        html.Img(src="/assets/ppg.png", className='pure-img-responsive'),
                                        ], className='post-images pure-g')
                                    ], className='post-description')
                            ],
                            className="post",
                        ),
                        # ===================== Post ===========================
                        # ===================== Post ===========================
                        html.Section(
                            [
                                html.Header([
                                     html.Img(className="post-avatar", src="/assets/mathis_avatar.png", style={'height':'48', 'width':'48'}),
                                     html.Div("Deep Q-Learnig", className="post-title"),
                                     html.P([
                                         "By ", html.A("Mathis", className="post-author", href="#"),
                                         " - ", html.A("Mines", className="post-category post-category-pure")
                                         ], className='post-meta'),
                                        ], className="post-header",
                                ),
                                html.Div([
                                    # TODO: Mettre ici la derniere image d'entrainement.
                                    html.Div([
                                        # Update 
                                        html.Img(src="/assets/code.png", className='pure-img-responsive'),
                                        ], className='post-images pure-g')
                                    ], className='post-description')
                            ],
                            className="post",
                        ),
                        # ===================== Post ===========================
                        html.H1("Problem Solution", className="content-subhead"),
                        # ===================== Post ===========================
                        html.Section(
                            [
                                html.Header([
                                     html.Img(className="post-avatar", src="/assets/audrey_avatar.png", style={'height':'48', 'width':'48'}),
                                     html.Div("Output", className="post-title"),
                                     html.P([
                                         "By ", html.A("Audrey", className="post-author", href="#"),
                                         " - ", html.A("Mines", className="post-category post-category-pure")
                                         ], className='post-meta'),
                                        ], className="post-header",
                                ),
                                dcc.Graph(id="fig1", className="post-description"),
                            ],
                            className="post",
                        ),
                        # ===================== Post ===========================
                    ],
                    className="posts",
                ),
            ]
        )
    ],
    className="content pure-u-1 pure-u-md-18-24",
)

app.layout = html.Div([sidebar, output,], id="layout", className="pure-g")


@app.callback(
    Output("data-store", "data"),
    [Input("my-button", "n_clicks")],
    [
        State("slider1", "value"),
        State("slider2", "value"),
        State("slider3", "value"),
        State("slider4", "value"),
        State("dropdown", "value"),
    ],
)
def run_episode(n_clicks, value1, value2, value3, value4, dropdown_value):
    if (
        (n_clicks is not None)
        & (value1 is not None)
        & (value2 is not None)
        & (value3 is not None)
        & (value4 is not None)
        & (dropdown_value is not None)
    ):

        # 1. Load agent & env
        env = SchEnv(stoch=False)
        env.from_json("info_2020_scheduling/Mathis/PraindeMine.json")
        env.reset()

        if dropdown_value == "PPO":
            agent = Agent_ppo_se(
                0.0005, len(env.action_space), len(env.observation_space), 0.97
            )
            agent.model_loading(
                name="info_2020_scheduling/Pierre/PPG_sans_entropy/model"
            )
        else:
            agent = Agent(env.observation_space.__len__(), env.action_space.__len__())
            agent.model_loading(name="info_2020_scheduling/Mathis/carry")

        # 2. Update Environnement avec les valeurs des sliders
        env.products["Farine"]["stock"] = value1
        env.products["Plastique"]["stock"] = value2
        env.products["PainDeMie"]["stock"] = 0
        env.products["PainDeMie"]["demand"] = 0
        env.products["PainDeMieSansCroute"]["stock"] = 0
        env.products["PainDeMieSansCroute"]["demand"] = 0
        env.products["PainDeMieEmballe"]["demand"] = value3
        env.products["PainDeMieEmballe"]["stock"] = 0
        env.products["PainDeMieSansCrouteEmballe"]["demand"] = value4
        env.products["PainDeMieSansCrouteEmballe"]["stock"] = 0

        # Run Episode
        history = []
        while not env.done:
            state = env.observation_space
            history.append(state)
            state = np.array([state])
            action = agent.get_best_action(state, rand=False)
            _, _, _, _, _ = env.step([action])

        history = pd.DataFrame(
            history,
            columns=[
                "Stock Farine",
                "Demand Farine",
                "Stock Plastique",
                "Demand Plastique",
                "Stock Pain de Mie",
                "Demand Pain de Mie",
                "Stock Pain de Mie sans Croute",
                "Demand Pain de Mie sans Croute",
                "Stock Pain de Mie Emballe",
                "Demand Pain de Mie Emballe",
                "Stock Pain de Mie sans Croute Emballe",
                "Demand Pain de Mie sans Croute Emballe",
                "Four",
                "Four on",
                "Decrouteur",
                "Decrouteur on",
                "Emballeur",
                "Emballeur on",
            ],
        )

        return history.to_dict("records")
    return {}


@app.callback(Output("fig1", "figure"), [Input("data-store", "data")])
def make_gantt_fig(data):

    fig = make_subplots(rows=7, cols=1, shared_xaxes=True, vertical_spacing=0.01)
    products = list(ENV["products"].keys())

    if data is not None:
        if len(data) > 0:
            history = pd.DataFrame().from_records(data)
            start = pd.to_datetime("today")
            dt = pd.Timedelta("2h")
            time = [start + dt * i for i in range(len(history))]
            jobs = []
            for m in ["Four", "Decrouteur", "Emballeur"]:
                for i in range(len(history)):
                    if history.iloc[i][f"{m} on"] == 1:
                        t1 = start + dt * i
                        t2 = start + dt * (i + 1)
                        p = products[history.iloc[i][m] - 1]
                        jobs.append(dict(Task=m, Start=t1, Finish=t2, Resource=p))
            gantt = ff.create_gantt(
                jobs, index_col="Resource", group_tasks=True, show_colorbar=True
            )
            for trace in gantt.data:
                fig.append_trace(trace, row=1, col=1)
            i = 2
            for c in history.columns:
                if "stock" in c.lower():
                    trace = go.Scatter(
                        x=time, y=history[c], fill="tozeroy", line_shape="hv", name=c
                    )
                    fig.append_trace(trace, row=i, col=1)
                    fig.layout[f"yaxis{i}"]["range"] = [0, 20]
                    fig.layout[f"yaxis{i}"]["autorange"] = False
                    i += 1
            i = 2
            for c in history.columns:
                if "demand" in c.lower():
                    if history[c].sum() > 0:
                        trace = go.Scatter(
                            x=time,
                            y=history[c],
                            line_shape="hv",
                            showlegend=False,
                            mode="lines",
                            line=dict(color="black", dash="dash"),
                        )
                        fig.append_trace(trace, row=i, col=1)
                    i += 1
            fig.update_layout(
                height=1000, xaxis_showticklabels=True, legend_orientation="h"
            )
            fig.layout["yaxis"]["ticktext"] = gantt.layout["yaxis"]["ticktext"]
            fig.layout["yaxis"]["tickvals"] = gantt.layout["yaxis"]["tickvals"]
            fig.update_xaxes(showticklabels=False) # hide all the xticks
            fig.update_xaxes(showticklabels=True, row=7, col=1)

    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
