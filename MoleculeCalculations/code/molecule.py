import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re


class State(object):

    def __init__(self,
                 name=None,
                 T0=None,
                 omega=None,
                 we=None,
                 we_xe=None,
                 Be=None,
                 alphae=None,
                 D0=None,
                 Re=None):

        self.name = name            # name designation of the state
        self.T0 = T0                # electronic energy offset (cm^-1)
        self.omega = np.abs(omega)  # omega quantum number (unsigned)
        self.we = we                # vibrational constant (cm^-1)
        self.we_xe = we_xe          # anharmonicity vibrational constant (cm^-1)

        if (Be is not None) and not(isinstance(Be,list)):
            Be = [Be]

        self.Be = Be                # rotational constant (list of 1 or 2)
        self.alphae = alphae        #
        self.D0 = D0                # centrifugal distortion
        self.Re = Re                # bond length in angstroms
        
        for key in self.__dict__.keys():
            try:
                if np.isnan(self.__dict__[key]):
                    self.__dict__[key] = None
            except:
                pass
            

    def _list_to_value(self,list_input,omega_doublet):

        if list_input is None:
            return None

        if not(isinstance(list_input,list)):
            return list_input

        if len(list_input) == 1:
            return list_input[0]

        if omega_doublet is None:
            return sum(list_input)/float(len(list_input))
        elif omega_doublet == 'e':
            return list_input[0]
        elif omega_doublet == 'f':
            return list_input[1]

    def __str__(self):
        return self.name + " state"

    def __repr__(self):
        return str(self)

    def energy(self, v=None, j=None, omega_doublet=None):
        """
        calculates the energy for a level with given quantum numbers

        :param v: vibrational quantum number (0 to inf integer)
        :param j: rotation quantum number (0-inf integer)
        :param omega_doublet: indicates upper or lower omega doublet ('e' (lower) or 'f' (upper))
        :return: energy in cm^-1
        """

        # first deal with the quantum numbers:
        if v is None:
            v = 0
        if j is None:
            if self.omega is None:
                j = 0
            else:
                j = np.abs(self.omega)
        else:
            if j < self.omega:
                raise IOError("j cannot be less than omega")

        total = 0
        if self.T0 is not None:
            total += self.T0
        if self.we is not None:
            total += self.we*v
        if self.we_xe is not None:
            total += -self.we_xe*((v+1/2.)**2-(1/2.)**2)
        if self.Be is not None:
            total += self._list_to_value(self.Be,omega_doublet)*j*(j+1)
        if self.D0 is not None:
            total += -self._list_to_value(self.D0,omega_doublet)*(j*(j+1))**2

        return total


class Molecule(object):

    def __init__(self,name=None, states=[], path=None):
        self.name = name
        self.states = {state.name: state for state in states}
        path = path

    def __str__(self):
        return str(self.name)

    def __repr__(self):
        return str(self)

    def _transition_type_to_quantum_nums(self,transition_type):
        if transition_type is None:
            return None
        if transition_type[0] == 'P':
            delta_j = -1
        elif transition_type[0] == 'Q':
            delta_j = 0
        elif transition_type[0] =='R':
            delta_j = +1
        else:
            raise IOError("onlt P,Q,R transitions are accepted")

        j1 = int(re.findall(r'\d+', transition_type)[0])
        j2 = j1 + delta_j
        return j1,j2

    def transition(self,
                   name1=None,v1=None,j1=None,omega_doublet1=None,
                   name2=None,v2=None,j2=None,omega_doublet2=None,
                   rot_transition=None,
                   transition_type=None,
                   numeric=False,
                   units='cm^-1'):
        """
        calculates the transition energy between two states in the molecule in cm^-1
        """
        if name1 is None or name2 is None:
            raise IOError("you must input two state names")

        if transition_type is not None:
            j1, j2 = self._transition_type_to_quantum_nums(transition_type)

        if not(name1 in self.states.keys()) or not(name2 in self.states.keys()):
            raise IOError("at least one input state is invalid")

        state1 = self.states[name1]
        state2 = self.states[name2]

        if rot_transition is not None:
            if rot_transition in ['R','Q','P']:
                if rot_transition == 'R':
                    delta_j = +1
                elif rot_transition == 'Q':
                    delta_j = 0
                elif rot_transition == 'P':
                    delta_j = -1

                if j1 is not None and j2 is not None:
                    raise IOError("""specifying rot_transition, j1, and j2 overconstrains the transition,
                                  please specify only one of the two j's when supplying rot_transition """)
                else:
                    if j1 is not None:
                        j2 = j1 + delta_j
                    if j2 is not None:
                        j1 = j2 - delta_j
            else:
                raise IOError("rot_transition must take values in ['R','P','Q']")

        energy_difference = (state2.energy(v=v2, j=j2, omega_doublet=omega_doublet2) -
                             state1.energy(v=v1, j=j1, omega_doublet=omega_doublet1))
        if numeric:
            return energy_difference
        else:
            ket1 = QuantumNumbers(name=name1, v=v1, j=j1, omega=omega_doublet1)
            ket2 = QuantumNumbers(name=name2, v=v2, j=j2, omega=omega_doublet2)
            return (str(ket1) + ' to ' + str(ket2) + ': ' + str(energy_difference) + ' cm^-1')

    def print_transition(self, **kwargs):
        energy_difference = self.transition(**kwargs)
        print energy_difference
        
    def to_dataframe(self):
        """
        convert the molecular states into a dataframe that can be visualized as a table.
        """
        #find all properties of all of the states:
        keys = []
        states = self.states.values()
        for state in states:
            keys = set(list(keys) + state.__dict__.keys())
        
        # create a dictionary to hold the data temporarily
        data = {key: [None] * len(self.states) for key in keys}
        
        # fill the dictionary
        for i in range(len(states)):
            for k, v in states[i].__dict__.items():
                data[k][i] = v
           
        # convert the dictionary to a dataframe
        df = pd.DataFrame.from_dict(data)
        df = df.set_index('name')
        
        preferred_order = ['omega', 'T0', 'we', 'we_xe', 'Be', 'D0', 'alphae', 'Re']
        # obtain an ordering consistent with the preferred ordering
        cols = df.columns.tolist()
        new_cols = []
        for col in preferred_order:
            if col in cols:
                new_cols.append(col)
        for col in cols:
            if col not in new_cols:
                new_cols.append(col)
        
        # reorder the columns
        df = df[new_cols]
        
        # sort by electronic state
        if 'T0' in new_cols:
            df = df.sort('T0')
        
        return df
    
    def save(self,path=None):
        form = '.pkl'
        df = self.to_dataframe()
        identifier = self.name
        if path is None:
            if self.path is None:
                path = identifier + form
            else:
                path = self.path
        else:
            path = path.split('.')
            if len(path) > 1:
                path = path[:-1]
            path = '.'.join(path)
            path = path + '---' + identifier + form
            
        df.to_pickle(path)
        return
        
    @staticmethod
    def load(path):
        name = path.split('.pkl')[0]
        name = name.split('---')[-1]
        name = name.split('/')[-1]
        name = name.split('\\')[-1]
        df = pd.read_pickle(path)
        return Molecule.from_dataframe(df, name=name, path=path)
    
    @staticmethod
    def from_dataframe(df, name=None, **kwargs):
        """
        generate a molecule object from a dataframe.
        """
        states = []
        for i in df.index:
            properties = dict(df.T[i])
            states.append(State(name=i, **properties))
    
        return Molecule(name=name, states=states, **kwargs)
    
    def energy_level_plot(self, states=None, transitions=None, decays=None,
                          v=4, width=.32, show=False, 
                          color=sns.color_palette()[0],
                          transition_color=sns.color_palette()[1],
                          decay_color=sns.color_palette()[2],
                          aspect_ratio = 1.618, font_size=14, **kwargs):
        """
        create an energy level diagram for the molecule
        """
        if states is None:
            states = self.states.keys()
        
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

        min_e = 0
        max_e = 0
        max_w = 0

        for state in self.states.values():
            if state.name in states:
                for v_i in range(v):
                    eng = state.energy(v=v_i)

                    x = [state.omega - width, state.omega + width]
                    y = [eng, eng]
                    ax.plot(x,y,'-', alpha=1/float(v_i + 1)**3, color=color, **kwargs)

                    if v_i == 0 and not(np.isnan(eng)):

                        min_e = min([min_e, eng])
                        max_e = max([max_e, eng])
                        max_w = max([max_w, state.omega])

                        plt.text(float(state.omega) + width + .15, eng, state.name,
                                va='center', ha='right', size=font_size)
                        
        if decays is None:
            decays = []
        if transitions is None:
            transitions = []
                        
        for decay in decays:
            tstates = [self.states[decay[0]], self.states[decay[1]]]
            x = [tstates[0].omega, tstates[1].omega]
            y = [tstates[0].energy(), tstates[1].energy()]
            ax.plot(x,y,'--',color=decay_color)#,alpha=.5)
        
        for transition in transitions:
            tstates = [self.states[transition[0]], self.states[transition[1]]]
            x = [tstates[0].omega, tstates[1].omega]
            y = [tstates[0].energy(), tstates[1].energy()]
            ax.plot(x,y,'-',color=transition_color)#,alpha=.75)

        y_range = max_e - min_e
        
        ax.set_xlim([-1.5*width, max_w + 1.5*width])
        ax.set_ylim([min_e - y_range/30, max_e + y_range/5])
        ax.set_xticks(list(range(max_w + 1)))

        ax.set_xlabel('$|\Omega|$')
        ax.set_ylabel('energy cm$^{-1}$')
        ax.set_title(self.name)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.set_aspect(.00009*aspect_ratio)

        if show:
            plt.show()
        return fig

class QuantumNumbers(object):

    def __init__(self,
                 name=None,
                 v=None,
                 j=None,
                 omega=None,
                 m=None,
                 Lambda=None,
                 Sigma=None):
        self.name = name
        self.v = v
        self.j = j
        self.omega = omega
        self.m = m
        self.Lambda = Lambda
        self.Sigma = Sigma

    def __str__(self):
        output = '|'
        parser = ', '
        separator = '; '
        if self.name is not None:
            output += self.name + parser
        if self.v is not None:
            output += 'v=' + str(self.v) + parser
        if self.j is not None:
            output += 'j=' + str(self.j) + parser
        if self.m is not None:
            output += 'm=' + str(self.m) + parser
        output = output[:-len(parser)] + separator
        if self.omega is not None:
            output += 'W=' + str(self.omega) + parser
        if self.Lambda is not None:
            output += 'L=' + str(self.Lambda) + parser
        if self.Sigma is not None:
            output += 'S=' + str(self.Sigma) + parser

        if output[-len(parser):] == parser:
            output = output[:-len(parser)]

        if output[-len(separator):] == separator:
            output = output[:-len(separator)]

        output += '>'

        return output

    def __repr__(self):
        return str(self)

