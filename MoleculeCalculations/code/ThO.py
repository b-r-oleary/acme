from molecule import State, Molecule

states = [State(name='X', T0=0,        omega=0, we=895.77, we_xe=2.39, Be=.33264,           alphae=.00130,           D0=1.83*10**-7,                Re=1.840),
          State(name='H', T0=5316.60,  omega=1, we=857.2,  we_xe=2.4,  Be=.32642,           alphae=.00129,           D0=1.891*10**-7,               Re=1.858),
          State(name='Q', T0=6127.921, omega=2, we=859.42, we_xe=2.29, Be=.326353,          alphae=.00133,           D0=1.919*10**-7,               Re=1.856),
          State(name='A', T0=10600.82, omega=0, we=846.4,  we_xe=2.4,  Be=.32304,           alphae=.00129,           D0=1.87*10**-7,                Re=1.867),
          State(name='B', T0=11129.14, omega=1, we=842.8,  we_xe=2.18, Be=[.32364, .32497], alphae=[.00129, .00130], D0=[1.88*10**-7, 1.94*10**-7], Re=1.864),
          State(name='C', T0=14490.00, omega=1, we=825.1,  we_xe=2.4,  Be=[.32161, .32246], alphae=[.00129, .00128], D0=[1.87*10**-7, 1.93*10**-7], Re=1.870),
          State(name='E', T0=16320.37, omega=0, we=829.26, we_xe=2.3,  Be=.32309,           alphae=.00130,           D0=1.99*10**-7,                Re=1.867),
          State(name='F', T0=18337.56, omega=0, we=757.36, we_xe=None, Be=.32140,           alphae=None,             D0=2.04*10**-7,                Re=1.870),
          State(name='G', T0=18009.93, omega=2, we=809.1,  we_xe=2.3,  Be=.31814,           alphae=.00128,           D0=1.94*10**-7,                Re=1.882),
          State(name='I', T0=19539.06, omega=1, we=800.85, we_xe=1.47, Be=[.32892, .33043], alphae=[.00191, .00183], D0=[2.21*10**-7, 2.39*10**-7], Re=1.849),
          State(name='M', T0=21734.32, omega=1, we=800.85, we_xe=None, Be=[.32575, .32586], alphae=[.00140, .00139], D0=[2.05*10**-7, 2.06*10**-7], Re=1.860),
          State(name='S', T0=20061.66, omega=3, we=None,   we_xe=None, Be=.315050,          alphae=None,             D0=1.987*10**-7,               Re=1.889),
          State(name='T', T0=24035.58, omega=3, we=None,   we_xe=None, Be=.316785,          alphae=None,             D0=1.994*10**-7,               Re=1.884),
          State(name='R', T0=19050.75, omega=0, we=862.0,  we_xe=2.8,  Be=.33232,           alphae=.00148,           D0=2.001*10**-7,               Re=1.841)
          ]

ThO = Molecule(name='ThO', states=states)