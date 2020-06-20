import os
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sch_env import SchEnv
from info_2020_scheduling.Mathis.train import *
from info_2020_scheduling.Mathis.agentsch import *
from info_2020_scheduling.Pierre.PPG_sans_entropy.manon_agent_ppg import *
from info_2020_scheduling.Pierre.PPG_sans_entropy.manon_exec_ppg import *
import json
import gym

env = SchEnv(stoch = False)
env.from_json("info_2020_scheduling/Mathis/PraindeMine.json")
agent_DQL = Agent(env.observation_space.__len__(), env.action_space.__len__())
agent_DQL.model_loading(name = "info_2020_scheduling/Mathis/carry")
agent_ppo_se = Agent_ppo_se(0.0005, len(env.action_space), len(env.observation_space), 0.97)
agent_ppo_se.model_loading(name = "info_2020_scheduling/Pierre/PPG_sans_entropy/model")
Name = None
with open("info_2020_scheduling/Mathis/PraindeMine.json", "r") as fichier :
    donnees= json.load(fichier)
    Name = list(donnees["products"].keys())


###CODE SCHEDULE
#machine allume ou eteint : env.machines["Four"]["is on"] : repond true ou false
#env.machines["Four"]["remain_setup"] : repond le temps qui reste en unite arbitraire (si c'est 0 la machine est on)
df = [dict(Task="Machine 1", Start='2020-06-01', Finish='2020-06-02', Resource='Flour'),
      dict(Task="Machine 2", Start='2020-06-02', Finish='2020-06-04', Resource='Sandwich bread'),
      dict(Task="Machine 3", Start='2020-06-03', Finish='2020-06-05', Resource='Cut sandwich bread'),
      dict(Task="Machine 4", Start='2020-06-03', Finish='2020-06-06', Resource='Cut and wrapped sandwich bread')]
colors = ['#7a0504', (0.2, 0.7, 0.3), 'rgb(210, 60, 180)', "#FF9966"]
fig = ff.create_gantt(df, colors=colors, index_col='Resource', reverse_colors=True,
                      show_colorbar=True)

###CODE HISTOGRAMME   
x = ['1', '2', '3', '4', '5',
     '6', '7']
y = [1,2,2,3,3,3,4]
fig2 = make_subplots(rows=2, cols=2)
# avec products liste de poduit, en y products.append((env.products["PaindeMieEmballe"]["stock"])

Product1 = go.Histogram(x=x, nbinsx=4, y=y)
Product2 = go.Histogram(x=x, nbinsx = 8)
Product3 = go.Histogram(x=x, nbinsx=10)
Product4 = go.Histogram(x=x, nbinsx = 3)

fig2.append_trace(Product1, 1, 1)
fig2.append_trace(Product2, 1, 2)
fig2.append_trace(Product3, 2, 1)
fig2.append_trace(Product4, 2, 2)


###CODE STOCK

fig3 = go.Figure(data=[go.Scatter(x=[1, 2, 3,4,5,6,7], y=[1,6,8,9,10, 7, 8])])
# y au moment ou on demande y.append(env.products["Farine"]["stock"])
#dans train.py il faut recopier le paragraphe dans valide while not env.done


###CODE PRE LAYOUT
app = dash.Dash(__name__)
server = app.server
colors = {
    'background': '#111111',
    'text': '#7FDBFF',
    "subtext": "#FF9900"
}

###LAYOUT
app.layout = html.Div(style={"backgroundColor": colors["background"]}, children=[
    dcc.Store(id='data-store'),
    html.H1('Welcome to our app', style= {"textAlign" : "center", "color" : colors["text"]}),
    html.H2("Optimize scheduling", style= {"texteAlign" : "center", "color" : colors["subtext"]}),
###INPUTS
    html.H2 ("Choose your inputs", style= {"textAlign" : "left", "color" : "#FFFFFF"}),
 
###ADRESSE MAIL         
    html.Div(id='display-value', style = {"backgroundColor" : "#111111"}),
    dcc.Input(id='my-id', value='Enter your email adress', type='text'),
    html.Div(id='my-div', style = {"backgroundColor" : "#111111", "color" : "#FFFFFF"}), 
    
###PRODUIT 1
    html.H3(Name[0], style= {"textAlign" : "left", "color" : colors["text"]}),
    dcc.Slider(id='slider1',min=0,max=50,step=1,value=10),
    html.Div(id='slider-output-container1', style = {"backgroundColor" : "#111111", "color": "#FFFFFF"}),
    
###PRODUIT 2
    #dcc.Input(id='my-id2', value='Enter a number', type='text'),
    #html.Div(id='my-div2', style = {"backgroundColor" : "#111111", "color" : "#FFFFFF"}),
    html.H3(Name[1], style= {"textAlign" : "left", "color" : colors["text"]}),
    dcc.Slider(id='slider2',min=0,max=50,step=1,value=10),
    html.Div(id='slider-output-container2', style = {"backgroundColor" : "#111111", "color": "#FFFFFF"}),

###PRODUIT 3
    #dcc.Input(id='my-id3', value='Enter a number', type='text'),
    #html.Div(id='my-div3', style = {"backgroundColor" : "#111111", "color" : "#FFFFFF"}),
    html.H3(Name[4], style= {"textAlign" : "justify", "color" : colors["text"]}),
    dcc.Slider(id='slider3',min=0,max=50,step=1,value=10),
    html.Div(id='slider-output-container3', style = {"backgroundColor" : "#111111", "color": "#FFFFFF"}),

#PRODUIT 4    
    #dcc.Input(id='my-id4', value='Enter a number', type='text'),
    #html.Div(id='my-div4', style = {"backgroundColor" : "#111111", "color" : "#FFFFFF"}),     
    html.H3(Name[5], style= {"textAlign" : "justify", "color" : colors["text"]}),
    dcc.Slider(id='slider4',min=0,max=50,step=1,value=10),
    html.Div(id='slider-output-container4', style = {"backgroundColor" : "#111111", "color": "#FFFFFF"}),
    
    
###ALGORITHME    
    #dcc.Input(id='my-id5', value='Enter a number', type='text'),
    #html.Div(id='my-div5', style = {"backgroundColor" : "#111111", "color" : "#FFFFFF"}),
    html.Label('Choose your algorithm'),
    html.H2('Choose your algorithm', style = {"backgroundColor" : "#111111", "color" : "#FFFFFF"}),
    dcc.Dropdown(id = "dropdown", options= [{'label':'PPO', 'value': 'PPO'}, {'label':'Deep Q-Learning', 'value': 'Deep Q-Learning'}], style={'backgroundColor': '#FFFFFF'}),
        
        
  
     
    #html.Div(dcc.Input(id='input-on-submit', type='text')),
    #html.Button('Submit', id='submit-val', n_clicks=0),
   # html.Div(id='container-button-basic', children='Enter a value and press submit'),
    
###BOUTON RUN   
   html.Button('RUN', id='my-button', n_clicks=0, style = {"backgroundColor": '#7FDBFF'}),
        
###OUTPUTS
    html.H2 ("Outputs", style= {"textAlign" : "left", "color" : "#FFFFFF"}),
###SCHEDULE
    html.H3("Machines' schedule", style= {"textaling" : "left", "color" : colors["text"]}), 
    dcc.Graph(id = "fig1"),
    
###HISTOGRAMME
    html.H3("Products' quantity", style= {"textaling" : "left", "color" : colors["text"]}),
     dcc.Graph(figure=fig2),
    
###STOCK    
    html.H3("Stock", style= {"textaling" : "left", "color" : colors["text"]}),
    #dcc.Graph(id='example-graph-2', figure=fig3)   

])





    
app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})


@app.callback(
        Output('data-store', 'data'),
        [Input('my-button', 'n_clicks')
            ],
        [
        State('slider1', 'value'),
        State('slider2', 'value'),
        State('slider3', 'value'),
        State('slider4', 'value'),
        State('dropdown', 'value'),
            ])
def run_episode(n_clicks, value1, value2, value3, value4, dropdown_value):
    env.products[Name[0]]["stock"] = value1
    env.products[Name[1]]["stock"] = value2
    env.products[Name[2]]["stock"] = value3
    env.products[Name[3]]["stock"] = value4

    #1. Load agent & env
    if dropdown_value == 'Deep Q-learning':
        agent = agent_DQL
    elif dropdown_value == 'PPO':
        agent = agent_ppo_se

    #2. Update Environnement avec les valeurs des sliders

    data = {}
    # Run Episode 
    # TODO: Update code bellow
    while not env.done:
        state = env.observation_space
        history.append(state)
        action = agent.get_best_action(state, rand = False)
        _, _, _, _, _ = env.step([action])
        
    history = pd.DataFrame(history, columns=['Stock_1', 'Demand_1',
                                           'Stock_2', 'Demand_2',
                                           'Stock_3', 'Demand_3',
                                           'Stock_4', 'Demand_4',
                                           'M1_process', 'M1_on',
                                           'M2_process', 'M2_on',
                                           'M3_process', 'M3_on',
                                            'M4_process', 'M4_on'])

    return history.to_dict('records')

            

@app.callback(
    Output('fig1', 'fig'),
    [Input('data-store', 'data')]
    )
def make_gantt_fig(data):

    history = pd.DataFrame().from_records(data)
   
    etat1 = list(history["M1_on"])
    etat2 = list(history["M2_on"])
    etat3 = list(history["M3_on"])
    etat4 = list(history["M4_on"])
    etat = [etat1, etat2, etat3, etat4]

    #on recupere les indices des listes ou is_on == False
    eteint = []
    allume = []
    for j in range(len(etat)):
        L=[] ; M=[]
        for i in range (len(etat[j]-1)):
            if (etat[j][i+1] == False) and (etat[j][i] == True) :
                L.append(i+1)
            elif (etat[j][i] == False) and (etat[j][i+1] == True) :
                M.append(i+1)
        eteint.append(L)
        allume.append(M)

    df = []
    # for j in range(len(eteint)):
    #     for i in range(len(eteint[0])):
    #         while (eteint[i+1] - eteint[i]) < 31:
    #             if allume[i] < 10:
    #                 date_debut = f"2020-06-0{allume[j][i]}"
    #             elif allume[i] >= 10 :
    #                 date_debut = f"2020-06-{allume[j][i]}"
    #             elif eteint[i]<10:
    #                 date_fin = f"2020-06-0{eteint[j][i]}"
    #             elif eteint[i]>=10:
    #                 date_fin= f"2020-06-{eteint[j][i]}"
    #         df.append(dict(Task= f"Machine{j}", Start = date_debut, Finish = date_fin))



 #   df = [dict(Task="Machine 1", Start='2020-06-01', Finish='2020-06-02', Resource='Flour'),
  #        dict(Task="Machine 2", Start='2020-06-02', Finish='2020-06-04', Resource='Sandwich bread'),
   #       dict(Task="Machine 3", Start='2020-06-03', Finish='2020-06-05', Resource='Cut sandwich bread'),
    #      dict(Task="Machine 4", Start='2020-06-03', Finish='2020-06-06', Resource='Cut and wrapped sandwich bread')]

    fig = ff.create_gantt(df, colors=colors, index_col='Resource', reverse_colors=True,
                          show_colorbar=True)

    return fig


@app.callback(
    Output('fig2', 'figure'),
    [Input('data-store', 'data')]
    )
def make_bar_plots(data):
    history = pd.DataFrame().from_records(data)
    figure, ax = plt.subplots(5,1, figsize=(15,15), sharex=True)
    n = len(Name)
    # for i in range(n):
    #     ax[i].bar(history.index, history[f"Stock_{i+1}"], color=f"C{i}", label=f"Stock Product {i+1}", width=0.9)
    #     ax[i].axhline(history.iloc[0][f"Demand_{i+1}"], color=f"C{i}", label=f"Demand for Product {i+1}")
    #     ax[i].legend(shadow=True, fancybox=True)
    #     ax[i].set_ylim(0,50)

    return figure


if __name__ == '__main__':
    app.run_server(debug=True)
