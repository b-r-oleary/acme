from scipy.linalg import expm
from types import FunctionType
from multiprocessing import Pool, cpu_count
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
sns.set_context('poster')


class IntegrateSchrodingerEquation(object):
    """
    this is a class used to set up a quantum dynamical system
    and then to integrate the schrodinger equation and report results
    """
    def __init__(self, t=None, dt=None, energy=None, 
                 omega=None, omega_structure=None, names=None, psi0=None,
                 auto_integrate=True, normalize_input_state=True, 
                 record_intermediate_states=True):
        """
        
        inputs:
        t: (numpy array)
        dt: (numpy array or float)
        energy: (list of complex floats, or list of numpy arrays)
        omega: (list of complex floats, or list of numpy arrays)
        omega_structure: (list of lists of length 2 with integers)
        
        the time array can be built in a number of ways.
        - if dt is input, and is an array, this is used to determine time steps
        - if dt is input, and is a float, then it is assumed that we have uniform time
          sampling with that input time.
        - if t is input, and is an array then dt = t[1:] - t[:-1]
        
        the energy list indicates the energies of the energy eigenstates when there is no omega,
        energies = [E_0, E_1, E_2, ...]
        If this is a list of arrays, then it is assumed that the energies are time dependent.
        energies = [E_0(t), E_1(t), E_2(t), ...]
        if the energy list is a dictionary, then we can use the keys to denote states rather than
        indices:
        energies = {'state a': E_a(t), 'state b': E_b(t), ...}
        
        similarly the omega list indicates the strength of couplings between the energy eigenstates
        omega = [omega_1, omega_2, ...]
        and these couplings can be time dependent if arrays are input:
        omega = [omega_1(t), omega_2(t), ...]
        if omega is given as a dictionary of form:
        omega = {(0, 1): omega_1(t), (0, 2): omega_2(t)}
        then this indicates that coupling omega_1(t) couples between energy eigenstates of
        index 0 and 1.
        
        instead of using the dictionary representation
        """   
        
        if psi0 is None:
            raise IOError('you must input an initial state for the integration')
        self.psi0 = psi0
        self.n_states = len(psi0)
        
        if energy is None:
            energy = [0] * self.n_states
        else:
            self.n_states = len(energy)
        #elif len(energy) != self.n_states:
        #    raise IOError('the length of the energy input list must have the same length as psi0')
        
        if omega is None:
            omega = []
            omega_structure = []
        
        #handle the case in which we use dictionaries as inputs.
        if isinstance(energy, dict):
            self.names = sorted(energy.keys())
            energy = [energy[k] for k in self.names]
        else:
            if names is not None:
                if len(names) != len(energy):
                    raise IOError('the length of the names list must be the same as that of the energy list')
                self.names = names
            else:
                self.names = list(range(len(energy)))
                
        self.psi0 = self.state_dict_to_array(self.psi0)
            
        if isinstance(omega, dict):
            omega_structure = []
            new_omega = []
            for key, value in omega.items():
                inds = []
                if len(key) != 2:
                    raise IOError('the keys to omega must be a list or tuple of length 2')
                omega_structure.append([self.names.index(key[0]), self.names.index(key[1])])
                new_omega.append(value)
            omega = new_omega
                
        else:
            if (len(omega) > 0) and (omega_structure is None):
                raise IOError('you must indicate the states that are coupled by the entries in omega (either in omega as a dictionary or in omega_structure as a list)')
            elif len(omega) != len(omega_structure):
                raise IOError('the length of omega and omega_structure must be the same')
            for i in range(len(omega_structure)):
                inds = omega_structure[i]
                omega_structure[i] = [self.names.index(inds[0]), self.names.index(inds[1])]
                
        self.omega_structure = omega_structure
        self.n_iter = None # this is the number of time samples to integrate over
        
        # handle different methods for inputting time arrays
        if (t is None) and (dt is None):
            raise IOError('you must input either a time array t, or a time difference array dt, or a time step dt.')
        
        if t is not None:
            try:
                t = np.array(t)
                dt = t[1:] - t[:-1]
                t  = (t[1:] + t[:-1])/2
            except:
                raise IOError('t must be an array of times')
        
        # try to figure out how many samples there are from the energy and omega inputs.
        self._update_n_iter(dt, list_variable=False)
        self._update_n_iter(energy)
        self._update_n_iter(omega)
        
        # handle the case that n_iter is not yet resolved
        if self.n_iter is None:
            raise IOError('you must input at least one time dependent array to set the number of iterations.')

        # expand all entries so that they are all time arrays
        self.dt = self._update_time_arrays(dt, list_variable=False)
        self.energy = self._update_time_arrays(energy)
        self.omega  = self._update_time_arrays(omega)
        
        # create a time variable if one does not yet exist
        if t is None:
            t = np.cumsum(self.dt) - self.dt[0]
            
        self.t = t

        # this will contain the record of the time evolution of psi:
        self.record_intermediate_states = record_intermediate_states
        self.psi_record = []
        
        # this will be the final state:
        self.psi1 = None
        
        # integrate the schrodinger equation
        if auto_integrate:
            self.integrate()
        
    def state_dict_to_array(self, state):
        if isinstance(state, dict):
            new_state = np.zeros(self.n_states, dtype='complex128')
            for k, v in state.items():
                new_state[self.names.index(k)] = v
        else:
            new_state = state
        return self.normalize_state(np.array(new_state))
        
    def normalize_state(self, state):
        return state/np.sqrt(np.dot(np.conj(state.T), state))
        
    def _update_n_iter(self, variable, list_variable=True):
        if not(list_variable):
            variable = [variable]
        for var in variable:
            try:
                if self.n_iter is None:
                    self.n_iter = len(var)
            except:
                pass
        return
        
    def _update_time_arrays(self, variable, list_variable=True):
        new_variable = []
        if not(list_variable):
            variable = [variable]
        for var in variable:
            try:
                if len(var) == self.n_iter:
                    new_variable.append(np.array(var))
                elif len(var) == self.n_iter + 1:
                    var = np.array(var)
                    new_variable.append((var[1:] + var[:-1]) / 2)
                else:
                    raise IOError('the input time dependent arrays have incompatible lengths')
            except:
                new_variable.append(np.array([var] * self.n_iter))
        if not(list_variable):
            new_variable = new_variable[0]
        return new_variable
        
    def __str__(self):
        outputs = [':' * 10 + 'SCHRODINGER EQUATION INTEGRATION' + ':' * 10,
                    'number of states:\t' + str(self.n_states),
                   'state names:\t\t' + str(self.names),
                   'number of time steps:\t' + str(self.n_iter),
                   'couplings:\t\t' + str([tuple([self.names[inds[0]], self.names[inds[1]]])
                                           for inds in self.omega_structure]),
                   'average time step:\t' + str(np.mean(self.dt))]
        return '\n'.join(outputs)
        
    def __repr__(self):
        return str(self)
        
    def hamiltonian(self, i):
        """
        method used to construct the hamiltonian matrix.
        
        inputs:
        i: (int) index of the time point at which to evaluate the Hamiltonian
        """
        # create the matrix with the appropriate diagonal
        H = np.diag(
                np.array([e[i] for e in self.energy], dtype='complex128')
            )

        omega = [w[i] for w in self.omega]
        for w, inds in zip(omega, self.omega_structure):
            H[inds[0], inds[1]] = w
            H[inds[1], inds[0]] = np.conj(w)
                
        return H
        
    def integrate(self):
        
        psi = np.array(self.psi0, dtype='complex128')
        self.psi_record = []
        
        for i in range(self.n_iter):
            if self.record_intermediate_states:
                self.psi_record.append(psi)         # record current state
            H = self.hamiltonian(i)             # construct the Hamiltonian
            U = expm(-1j * H * self.dt[i])      # construct the time evolution operator
            psi = np.dot(U, psi)                # time evolve the state
                
        self.psi1 = psi
        if self.record_intermediate_states:
            self.psi_record = np.array(self.psi_record)
            
        return self.psi1
    
    def norm(self):
        norm = np.array(
                        [np.dot(self.psi_record[i,:], np.conj(self.psi_record[i, :])) 
                                                          for i in range(self.n_iter)]
                       )
        return norm
    
    def plot(self, states=None, names=None, plot_norm=True, 
             plot_states=True, normalize_input_states=True):
        if plot_states:
            if not(self.record_intermediate_states):
                self.record_intermediate_states=True
                self.integrate()
            
            if states is None:
                states = np.eye(self.n_states)
                if names is None:
                    names  = self.names
                elif len(names) != len(self.n_states):
                    raise IOError('the length of the names list must be same as number of states')

            if isinstance(states, dict):
                names = states.keys()
                states = states.values()
                
            for i in range(len(states)):
                states[i] = self.state_dict_to_array(states[i])

            if names is None:
                names = ['input state ' + str(i) for i in range(len(states))]

            states = np.array(states)
            
            if normalize_input_states:
                states = np.array(states)
                states = np.array([state/np.sqrt(np.dot(state, np.conj(state.T))) for state in states])

            overlap = np.abs(np.dot(self.psi_record, np.conj(states.T)))**2

            for i in range(len(states)):
                plt.plot(self.t, overlap[:,i], label=names[i])
            
        if plot_norm:
            norm = self.norm()
            plt.plot(self.t, norm, '--', label='norm')
            
        plt.xlabel('time (s)')
        plt.ylabel('probability')
        plt.legend(loc='best')
        return
        
    def plot_omega(self):
        names = [str(self.names[ind[0]]) + '-' + str(self.names[ind[1]])
                 for ind in self.omega_structure]
        for w, name in zip(self.omega, names):
            plt.plot(self.t, w, label=name)
            
        plt.xlabel('time (s)')
        plt.ylabel('rabi frequency\n($2\pi\cdot\mathrm{MHz}$)')
        plt.legend(loc='best')
        
    @staticmethod
    def parameter_sweep(values, 
                        parallelize=None, processes=None, final_state=False, 
                        **kwargs):
        """
        Performs a parameter sweep through parameter values given as a list *values*
        This function accepts the usual keyword inputs that IntegrateSchrodingerEquation accepts, 
        and any of those parameters may be replaced by functions that accept an argument of the form
        of an item in values to vary parameters.
        
        Additional inputs:
        parallelize (boolean) whether or not to parallelize the sweep
        processes (integer) number of cores to use for the sweep
        final_state (boolean) if true, returns only the final state, not the entire ISE object.
        """
        
        new_kwargs = [{k: (v(value) if isinstance(v, FunctionType) else v)
                                                for k, v in kwargs.items()}
                                                        for value in values]
        
        if parallelize is None:
            parallelize = (len(values) >= 20) 
        
        if final_state:
            integrator = _ise_return_only_final_state
        else:
            integrator = _ise
            
        if len(values) == 1:
            parallelize = False
        
        if parallelize:
            if processes is None:
                processes = min([len(values), cpu_count()])

            pool = Pool(processes=processes)
            output = pool.map(integrator, new_kwargs)
            pool.terminate()
        else:
            output = [integrator(kwargs) for kwargs in new_kwargs]
            
        return output
    

def _ise(kwargs):
    return IntegrateSchrodingerEquation(**kwargs)

def _ise_return_only_final_state(kwargs):
    return IntegrateSchrodingerEquation(**kwargs).psi1

