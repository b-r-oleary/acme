from sympy.physics.wigner import wigner_3j, wigner_6j, \
                                 wigner_9j, clebsch_gordan
from sympy import sqrt

def format_output(output, **kwargs):
    """
    i am using sympy functions to evaluate the wigner_nj symbols.
    sympy is a symbolic computing package and hence provides
    symbolic outputs. This function converts the output to a float
    if n=True is provided as a keyword argument.
    """
    for k, v in kwargs.items():
        if k == 'n':
            if v and output is not 0:
                if not(isinstance(output, int), 
                       isinstance(output, float)):
                    output = output.n()   
    return output

def wigner_eckart(j, m, k, q, j1, m1, n=True):
    """
    the wigner-eckart theorem allows extraction of the 
    angular dependence of a matrix element leaving the reduced
    matrix element that is no longer dependent on spatial
    orientation (M quantum numbers).
    
    See Brown and Carrington 5.172
    """
    output = (-1)**(j - m) * wigner_3j(j, k, j1, -m, q, m1)
    return format_output(output, n=n)
    
def spectator(a, b, j, k, a1, b1, j1, n=True):
    """
    assuming j = a + b is an angular momentum addition rule,
    and the operator is of spherical vector order k, and 
    if the operator does not act on the b angular moment,
    then remove the effect of b so that the effect may be evaluated
    on a.
    
    See Brown and Carrington 5.174
    """
    if b != b1:
        return 0
    else:
        output = ((-1)**(j1 + a + k + b) * 
                  sqrt((2 * j + 1) * (2 * j1 + 1)) *
                  wigner_6j(a1, j1, b, j, a, k))
        return format_output(output, n=n)
         
def reduced_matrix_element(j, n=True):
    """
    for a rank 1 spherical tensor operator, this is the
    reduced matrix element,
    <J||T^1||J1> (which vanishes for J1 != J, so there is
    only one input to this function)
    
    See Brown and Carrington 5.179
    """
    output = sqrt(j * (j + 1) * (j + 2))
    return format_output(output, n=n)
    
def reduced_lab_to_mol(j, w, k, Q, j1, w1, n=True):
    """
    This function describes the transformation from the
    laboratory frame to the molecular-axis-fixed frame.
    This assumes that the angular dependence has already
    been extracted using the wigner eckart theorem.
    
    see Brown and Carrington 5.186
    """
    output = ((-1)**(j - w) * wigner_3j(j, k, j1, -w, Q, w1) *
              sqrt((2 * j + 1) * (2 * j1 + 1)))
    return format_output(output, n=n)

def lab_to_mol(j, m, w, k, q, Q, j1, m1, w1, n=True):
    """
    this is the full transformation from the laboratory frame
    to the molecular-axis-fixed frame.
    
    see Brown and Carrington 5.185 and 5.186
    """
    output = ((-1)**(j - w) * wigner_eckart(j, m, k, q, j1, m1) *
              reduced_lab_to_mol(j, w, k, Q, j1, w1))
    return format_output(output, n=n)

def tensor_product(a, b, j, 
                   k1, k2, k, 
                   a1, b1, j1, 
                   n=True, precision=None):
    """
    If you have a matrix element of a product of two tensor operators
    of rank k1 and k2, this product operator has a spherical tensor
    operator component of rank k.
    
    This function describes the reduced matrix element:
    <J,A,B||(T^k1(A) T^k2(B))^k||J1,A1,B1>
    
    see Brown and Carrington 5.169
    """
    
    output = (sqrt((2 * j + 1) * (2 * k + 1) * (2 * j1 + 1)) *
              wigner_9j(a1, b1, j1, 
                        k1, k2, k, 
                        a,  b,  j, 
                        precision=precision))
    
    return formated_output(output, n=n)

def hunds_case_b_to_hunds_case_a(k, Q, 
                                 J1, S1, N1, lam1, sig1,
                                 J,  S,  N,  lam,  sig,
                                 n=True):
    """
    describes the transformation between the Hunds case (a)
    basis set and the Hunds case (b) basis set. Here, we must 
    implicitly sum over all allowed values of sig and sig1
    
    see Brown and Carrington 6.149
    """
    
    output = ((-1)**(J + J1 - S - S1 + lam + lam1) * 
              sqrt((2 * N + 1) * (2 * N1 + 1)) *
              wigner_3j(J1, S1, N1, lam1 + sig1, -sig1, -lam1) *
              wigner_3j(J,  S,  N,  lam  + sig,  -sig,  -lam)  *
              reduced_lab_to_mol(J, lam + sig, k, Q, J1, lam1 + sig1))
    
    return formated_output(output, n=n)