#! /usr/bin/env python

# Import the classes needed from sympy
from sympy import Tuple, Symbol, IndexedBase, Idx, Function, Expr, Eq, Derivative

# Import the functions needed from sympy
from sympy import summation, integrate, simplify, preorder_traversal

################################################################################

def expr_as_tuple(expr, fact=1):
    """
    Given a sympy.Expr object, return a tuple of sympy.Expr objects such that
    the sum of the tuple items equals the orginal sympy.Expr object
    """
    result = []
    if len(expr.args) == 0:
        result.append(fact * expr)
    else:
        for arg in expr.args:
            result.append(fact * arg)
    return tuple(result)

################################################################################

def Eq_as_tuple(eq):
    """
    Given a sympy.Eq equation, return a tuple that consists of all the terms of
    the left-hand side of eq, and the negative of all the right-hand side terms
    of eq.
    """
    result = []
    lhs = expr_as_tuple(eq.args[0])
    result.extend(lhs)
    rhs = expr_as_tuple(eq.args[1], -1)
    result.extend(rhs)
    return tuple(result)

################################################################################

def integrate_by_parts(u,v,x):
    """
    Given a user-defined u(x) and a user-defined v(x), return the three terms of
    the integration-by-parts formula in a tuple, such that the sum of the three
    terms equals zero.
    """
    # Store the derivative of u*v
    duv = Derivative(u*v,x)
    # Apply the chain rule to duv
    terms = duv.doit().args
    # Return the three terms of the integration-by-parts formula.
    # The sum of these three terms equals zero
    return (integrate(duv,x),
            -integrate(terms[0],x),
            -integrate(terms[1],x))

################################################################################

def weak_form(eqs, v):
    """
    Given one or more sympy.Eq objects that represents a set of partial
    differential equations and test function v(x), return a tuple of sympy.Expr
    that represents the weak form of the equation(s).

    Arguments:
        eqs  - (in) partial differential equation(s) (either sympy.Eq, or a
               tuple of sympy.Eq)
        v    - (in) test function (sympy.Function)

    Returns:
        The weak form of eqs, expressed as an expression that evaluates to zero,
        i.e. in the form of lhs - rhs = 0. If eqs is a single equation, then the
        result is a sympy.Expr.  If eqs is a tuple of sympy.Eq, then the result
        is a tuple of sympy.Expr
    """

    # Inspect the set of PDEs
    if isinstance(eqs, Eq):
        eqs = (eqs, )
    insp   = inspect_eqns(*eqs)
    num_eq = len(eqs)

    # Figure out the domain variable
    domain_vars = insp["domain variables"]
    if len(domain_vars) == 0:
        raise ValueError("No domain variables!")
    if len(domain_vars) > 1:
        raise ValueError("Only 1D problems supported. So far!")
    x = domain_vars[0]

    # Obtain the integrals of the PDEs * test function
    result = [0] * num_eq
    for i in range(num_eq):
        eq = eqs[i]
        eq_tuple = Eq_as_tuple(eq)
        for term in eq_tuple:
            result[i] += integrate(term*v,x).doit()

    # Use integration by parts to remove second derivatives
    sol_vars = insp["differentiated functions"]
    for i in range(num_eq):
        for u in sol_vars:
            ibp = integrate_by_parts(Derivative(u,x), v, x)
            o2_term  = ibp[2]
            o1_terms = -ibp[0] - ibp[1]
            result[i] = result[i].subs(o2_term, o1_terms)
        # atteach the inspection to each result
        # result[i].inspection = insp

    # Return the results
    if num_eq == 1:
        result = result[0]
    else:
        result = tuple(result)
    return result

################################################################################

def extract_funcs(expr):
    """
    Given a sympy expression, return a tuple of all the functions contained
    within that expression.
    """
    functions = set()
    for arg in preorder_traversal(expr):
        if isinstance(arg, Function):
            functions.add(arg)
    return tuple(functions)

################################################################################

def cleanup_inspection(insp):
    """
    Given a 'raw' equation inspection, remove from the set of symbols any
    symbols that also appear as a differentiating variable or a domain
    variable.  Also remove from the set of functions any functions that also
    appear as a differentiated function.

    Argument:
        insp    - (in/out) The inspection data of an equation
    """

    # Remove duplicates from the symbols set
    symbols     = list(insp["symbols"                  ])
    diff_vars   = set( insp["differentiating variables"])
    domain_vars = set( insp["domain variables"         ])
    for var in diff_vars.union(domain_vars):
        if var in symbols:
            symbols.remove(var)

    # Remove duplicates from the functions set
    functions  = list(insp["functions"])
    diff_funcs = insp["differentiated functions"]
    for func in diff_funcs:
        if func in functions:
            functions.remove(func)

    # Update inspection dictionary
    insp["symbols"  ] = tuple(symbols  )
    insp["functions"] = tuple(functions)

################################################################################

def inspect_eqn(eq):
    """
    Given a sympy Eq or Expr object, inspect it and return a dictionary that
    specifies sets of data about that expression.

    Argument:
        eq    - (in) a sympy equation object (sympy.Eq) or expression object
                (sympy.Expr)

    Returns:
        dict with keys:
            "symbols"                   - symbols that are not differentiating variables
            "functions"                 - functions that are not differentiated
            "differentiated functions"  - functions that are differentiated
            "differentiating variables" - variables that appear in the
                                          denominator of derivatives
            "domain variables"          - the set of all domain variables of
                                          functions and differentiated functions
    """

    # Initialize the equation data sets
    symbols     = set()
    functions   = set()
    diff_funcs  = set()
    diff_vars   = set()
    domain_vars = set()

    # Traverse the equation tree and fill in the data sets
    for arg in preorder_traversal(eq):
        if isinstance(arg, Symbol):
            symbols.add(arg)
        #if isinstance(arg, IndexedBase):
        #    symbols.add(arg)
        if isinstance(arg, Function):
            functions.add(arg)
            for func_arg in arg.args:
                domain_vars.add(func_arg)
        if isinstance(arg, Derivative):
            numerator    = arg.args[0]
            denominators = arg.args[1:]
            for term in extract_funcs(numerator):
                diff_funcs.add(term)
            if isinstance(denominators, tuple):
                for denominator in denominators:
                    if isinstance(denominator, Tuple):
                        diff_vars.add(denominator[0])
                    else:
                        diff_vars.add(denominator)
            else:
                diff_vars.add(denominators)

    # Convert data sets to tuples and store as a dictionary
    result = {"symbols"                   : tuple(symbols    ),
              "functions"                 : tuple(functions  ),
              "differentiated functions"  : tuple(diff_funcs ),
              "differentiating variables" : tuple(diff_vars  ),
              "domain variables"          : tuple(domain_vars)}

    # Clean up duplicates in the inspection and return
    cleanup_inspection(result)
    return result

################################################################################

def combine_inspections(insp1, insp2):
    """
    Given two equation inspections, combine them and clean the results up
    according to the rules of function cleanup_inspection()
    """

    # Initialize the result dictionary
    result = {}
    for key in insp1.keys():
        result[key] = tuple(set(insp1[key]).union(insp2[key]))

    # Clean up the inspection dictionary and return
    cleanup_inspection(result)
    return result

################################################################################

def inspect_eqns(*args):
    """
    Given a set of sympy.Eq or sympy.Expr objects, inspect them as a set and
    return a dictionary that specifies sets of data about that set of equations
    or expressions.

    Argument:
        *args    - (in) a set of sympy equation (sympy.Eq) or expression objects 
                   (sympy.Expr)

    Returns:
        dict with keys:
            "symbols"                   - symbols that are not differentiating variables
            "functions"                 - functions that are not differentiated
            "differentiated functions"  - functions that are differentiated
            "differentiating variables" - variables that appear in the
                                          denominator of derivatives
            "domain variables"          - the set of all domain variables of
                                          functions and differentiated functions
    """

    # Initialize an inspection with empty values
    result = {"symbols"                   : tuple(),
              "functions"                 : tuple(),
              "differentiated functions"  : tuple(),
              "differentiating variables" : tuple(),
              "domain variables"          : tuple()}

    # Loop over the arguments, inspect each one, and combine that inspection
    # with the existing insepction
    for arg in args:
        result = combine_inspections(result, inspect_eqn(arg))

    # Return the results
    return result

################################################################################

def galerkin(expr, test_func, basis):
    """
    Given a set of sympy.Expr objects, return a new set of sympy.Expr objects in
    which substitutions have been made that are consistent with a Galerkin
    approximation.

    Arguments:
        expr         - (in) a single sympy.Expr object, or a sequence of
                       sympy.Expr objects
        test_func    - (in) it is assumed that the expr are expressions obtained
                       from the weak_form() function, using the same test
                       function specified here
        basis        - (in) a sympy.Function object that represents the basis
                       functions for the Galerkin approximation.  Should be
                       constructed without arguments: this function will add
                       them

    Returns:
        one or more sympy.Expr objects
            A single expression will be returned as a sympy.Expr; multiple
            expressions wil be returned as a tuple of sympy.Expr
    """

    # Ensure exprs is always a tuple
    exprs = expr
    if isinstance(exprs, Expr):
        exprs = [exprs]
    else:
        exprs = list(expr)
    num_expr = len(exprs)

    # Define our indexes
    i = Idx('i')
    j = Idx('j')
    N = Symbol('N', integer=True)

    # Obtained the non-test differentiated functions
    insp = inspect_eqns(*exprs)
    diff_funcs = list(insp["differentiated functions"])
    try:
        diff_funcs.remove(test_func)
    except ValueError:
        pass

    # Obtain the domain variables
    x = insp["domain variables"]
    if len(x) == 1:
        x = x[0]
    else:
        raise ValueError("Only 1D problems supported. So far!")

    # Apply the Galerkin approximation
    for k in range(num_expr):
        exprs[k] = exprs[k].subs(test_func, basis(j,x))
        for func in diff_funcs:
            name   = str(func)
            paren  = name.find('(')
            name   = name[:paren]
            func_i = IndexedBase(name)[i]
            exprs[k] = exprs[k].subs(func, summation(func_i * basis(i,x), (i,0,N-1)))
        exprs[k] = exprs[k].doit()

    # Return the results
    if num_expr == 1:
        exprs = exprs[0]
    else:
        exprs = tuple(exprs)
    return exprs
