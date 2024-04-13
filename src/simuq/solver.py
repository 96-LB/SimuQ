"""
The solver of SimuQ.

The solver contains three steps:
(1) find an alignment of target sites and machine sites.
(2) synthesize the target Hamiltonian with instructions
without considering mutually exclusiveness.
(3) trotterize the resulting boxes of instructions.

The detailed descriptions are left in the implementations.
"""

import logging

import networkx as nx
import numpy as np

from simuq.expression import Expression
from simuq.hamiltonian import TIHamiltonian, productHamiltonian

logging.basicConfig(level=logging.CRITICAL)

logger = logging.getLogger()


# Functions that locate a local variable.
def locate_switch_evo(mach, evo_index):
    return len(mach.gvars) + evo_index * (mach.num_inss + len(mach.lvars))

def locate_nonswitch_evo(mach, evo_index):
    return len(mach.gvars) + evo_index * len(mach.lvars)

def locate_nonswitch_timevar(mach, evo_index, total_evo_num):
    return len(mach.gvars) + total_evo_num * len(mach.lvars) + evo_index


# Functions to locate local variables.
def locate_evo(mach, evo_index):
    return len(mach.gvars) + evo_index * (mach.num_inss + len(mach.lvars))


def locate_lvar(mach, evo_index, lvar_index):
    return locate_evo(mach, evo_index) + mach.num_inss + lvar_index

def locate_timevar(mach, total_evo_num, evo_index):
    return len(mach.gvars) + total_evo_num * (mach.num_inss + len(mach.lvars)) + evo_index


def ins_fun(mach, evo_index, ins_index, mc, total_evo_num):
    return lambda x: x[locate_timevar(mach, total_evo_num, evo_index)] * (
        x[locate_evo(mach, evo_index) + ins_index]
        * mc.exp_eval(
            x[: len(mach.gvars)],
            x[locate_evo(mach, evo_index) + mach.num_inss : locate_evo(mach, evo_index + 1)],
        )
    )

def locate_switch(mach, evo_index, ins_index):
    return locate_switch_evo(mach, evo_index) + ins_index


# Equation builder and solver when an alignment is supplied.
def solve_aligned(
    ali, qs, mach, solver="least_squares", solver_args={"tol": 1e-3, "time_penalty": 0}, verbose=0
):
    if isinstance(solver_args, float):
        tol = solver_args
        time_penalty = 0
    else:
        tol = solver_args["tol"]
        if "time_penalty" in solver_args.keys():
            time_penalty = solver_args["time_penalty"]
        else:
            time_penalty = 0
    logger.info(ali)
    if verbose > 0:
        print("Start_solving for ", ali)

    """
    Build equations. The equation variables are the global variables
    and copies of local variables (# evolution of copies).

    Each product term of the target Hamiltonian is used to build an
    equation, as the constant side. The other side contains all the
    possible contributor of this product in the machine's instructions.

    Since a instruction is either selected or not, we represent it
    by a switch variable for each instruction. Then it turns to a mixed
    equation solving problem ({0, 1} variables and continuous varibales).

    Currently, this is solved by first treating the switch variables as
    continuous variables, truncating them and solving the equations
    again with switch variables set to 0 or 1.
    """

    global gsol
    global gswitch
    global gtime
    global alignment

    if solver == "least_squares":
        
        
        import scipy.optimize as opt
        
        assert len(mach.gvars) == 0
        assert not mach.with_sys_ham
        assert len(qs.evos) == 1
        
        # first hamiltonian
        ham, gtime = qs.evos[0]
        targets = [(term, coefficient) for term, coefficient in ham.ham]
        def _get(term):
            for i in range(len(targets)):
                if targets[i][0] == term:
                    return targets.pop(i)[1]
            return 0
        
        gsol = [None] * len(mach.lvars)
        gswitch = []
        
        for line in mach.lines:
            gswitch.append([])
            for ins in line.inss:
                switch = 0
                eqns = []
                vars = set()
                for term, coefficient in ins.h.ham:
                    target = _get(term)
                    eqns.append((coefficient, target))
                    vars = vars.union(coefficient.vars)
                    if abs(target) > tol:
                        switch = 1
                vars = list(vars)
                
                gswitch[-1].append(switch)
                if switch:
                    map = {var: vars.index(var) for var in vars}
                    
                    def f(x):
                        result = np.zeros(len(eqns))
                        for i in range(len(eqns)):
                            x_ = [x[map[v]] for v in eqns[i][0].vars]
                            result[i] = eqns[i][0].exp(x_) - eqns[i][1]
                        return result
                    
                    x = opt.root(f, np.zeros(len(vars)))
                    if not x.success:
                        raise Exception("Failed to solve the equation.")
                    vals = x.x
                else:
                    vals = np.zeros(len(vars))
                
                for i in range(len(vars)):
                    if gsol[vars[i].index] is not None:
                        raise Exception('uhhh')
                    gsol[vars[i].index] = vals[i]
        
        alignment = ali
        gsol = np.array(gsol)
        gtime = [gtime]
        gswitch = [gswitch]
        
        return True
    
    else :
        raise Exception("Solver does not exist.")

# If the system has no global variable nor system
# Hamiltonian, then we can solve it piece by piece.
def solve_aligned_wrapper(ali, qs, mach, solver, solver_args, verbose):
    if mach.gvars or mach.with_sys_ham:
        return solve_aligned(ali, qs, mach, solver, solver_args, verbose)
    else:
        switch = []
        sol = np.zeros(len(mach.lvars) * len(qs.evos))
        time = []
        evos = qs.evos
        global gswitch
        global gsol
        global gtime
        for evo_index in range(len(evos)):
            qs.evos = [evos[evo_index]]
            if not solve_aligned(ali, qs, mach, solver, solver_args, verbose):
                return False
            switch.append(gswitch[0])
            sol[
                locate_nonswitch_evo(mach, evo_index) : locate_nonswitch_evo(mach, evo_index + 1)
            ] = gsol
            time.append(gtime[0])
        qs.evos = evos
        gswitch = switch
        gsol = sol
        gtime = time
        return True


"""
The search of a valid alignemnt.

Optimization used here: if the current alignment dooms a
non-zero product term in the target Hamiltonian, then break.
"""


def align(i, ali, qs, mach, solver, solver_args, verbose):
    def is_id(tprod):
        ret = True
        for i in range(qs.num_sites):
            if not tprod[i] == "":
                ret = False
                break
        return ret

    if i == qs.num_sites:
        if solve_aligned_wrapper(ali, qs, mach, solver, solver_args, verbose):
            return True
        return False
    for x in range(mach.num_sites):
        available = True
        ali[i] = x
        for j in range(i):
            if ali[j] == x:
                available = False
                break
        if not available:
            continue
        for h, t in qs.evos:
            for tprod, tc in h.ham:
                if is_id(tprod):
                    continue
                found = False
                for line in mach.lines:
                    for ins in line.inss:
                        for mprod, mc in ins.h.ham:
                            prod_partial_match = True
                            for k in range(i + 1):
                                if tprod[k] != mprod[ali[k]]:
                                    prod_partial_match = False
                            if prod_partial_match:
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break
                if not found:
                    available = False
                    break
            if not available:
                break
        if available:
            if align(i + 1, ali, qs, mach, solver, solver_args, verbose):
                return True
    return False


def find_sol(qs, mach, ali=None, solver="least_squares", solver_args=None, verbose=0):
    solver_args = solver_args or {"tol": 1e-2}
    if ali == []:
        return align(0, [0] * qs.num_sites, qs, mach, solver, solver_args, verbose)
    else:
        return solve_aligned_wrapper(ali, qs, mach, solver, solver_args, verbose)


def write_default(args, default_args):
    for k in default_args:
        if k not in args:
            args[k] = default_args[k]


# The generation of abstract schedule
# Trotterize the solution provided by the second step.
# Arg solver_tol will be deprecated. Please avoid its usage and use solver_args.
# Arg trotter_num will be deprecated. Please avoid its usage and use solver_args.
def generate_as(
    qs,
    mach,
    trotter_num=None,
    trotter_args=None,
    solver="least_squares",
    solver_tol=None,
    solver_args=None,
    override_layout=None,
    verbose=0,
):
    default_solver_args = {"tol": 1e-1, "with_time_penalty": False}
    solver_args = solver_args or default_solver_args
    if solver_tol is not None:
        solver_args["tol"] = solver_tol
    write_default(solver_args, default_solver_args)

    # "sequential" means whether temporal graph is sequential.
    # If True, the solver will generate parallel blocks for each Trotter step
    default_trotter_args = {"num": 6, "order": 1, "sequential": False}
    if trotter_args == None:
        trotter_args = default_trotter_args
    if trotter_num is not None:
        trotter_args["num"] = trotter_num
    write_default(trotter_args, default_trotter_args)

    if trotter_args["order"] > 2:
        raise Exception("Trotter order > 2 is not supported.")

    ali = [] if override_layout is None else override_layout
    mach.instantiate()
    mach.extend_instruction_sites()
    if find_sol(qs, mach, ali=ali, solver=solver, solver_args=solver_args, verbose=verbose):
        sol = gsol
        switch = gswitch

        sol_gvars = sol[: len(mach.gvars)]
        sol_time = gtime
        boxes = []
        edges = []
        ending_boxes = []

        # Deal with machines with system Hamiltonian first.
        if mach.with_sys_ham:
            for evo_index in range(len(qs.evos)):
                if len(sol) > 0:
                    sol_lvars = sol[
                        locate_nonswitch_evo(mach, evo_index) : locate_nonswitch_evo(
                            mach, evo_index + 1
                        )
                    ].tolist()
                else:
                    sol_lvars = []
                box = ([], sol_time[evo_index])
                for i in range(len(mach.lines)):
                    line = mach.lines[i]
                    for j in range(len(line.inss)):
                        ins = line.inss[j]
                        if switch[evo_index][i][j] == 1 and not ins.is_sys_ham:
                            ins_lvars = []
                            for l in range(len(ins.vars_index)):
                                ins_lvars.append(sol_lvars[ins.vars_index[l]])
                            box[0].append(
                                (
                                    (i, j),
                                    ins,
                                    ins.exp_eval(sol_gvars, sol_lvars),
                                    ins_lvars,
                                )
                            )
                if box:
                    boxes.append(box)

            logger.info(alignment)
            logger.info(sol_gvars)
            for box in boxes:
                logger.info(
                    (
                        [((i, j), ins_lvars) for ((i, j), ins, h, ins_lvars) in box[0]],
                        box[1],
                    )
                )
            edges = [(i, i + 1) for i in range(len(qs.evos) - 1)]
            for idx in range(len(qs.evos) - 1):
                logger.info((idx, idx + 1))

            return (alignment, sol_gvars, boxes, edges)

        # Then deal with other cases
        for evo_index in range(len(qs.evos)):
            next_ending_boxes = []
            coloring = [i for i in range(mach.num_sites)]
            sol_lvars = sol[
                locate_nonswitch_evo(mach, evo_index) : locate_nonswitch_evo(mach, evo_index + 1)
            ].tolist()
            # Detach by touched sites
            """
            Note that we assume that a derived instruction only
            affects those sites it touches (it commutes with any
            instruction not touching these sites). We can separate
            the instructions in the box by the sites they touch.

            Then we use a connectivity coloring to separate them first.
            """
            first_touch = [[-1 for i in range(len(line.inss))] for line in mach.lines]
            for i in range(len(mach.lines)):
                line = mach.lines[i]
                for j in range(len(line.inss)):
                    ins = line.inss[j]
                    if switch[evo_index][i][j] == 1:
                        t_sites = (ins.h.exp_eval(sol_gvars, sol_lvars)).touched_sites()
                        for k in range(mach.num_sites):
                            if t_sites[k] == 1:
                                if first_touch[i][j] == -1:
                                    first_touch[i][j] = k
                                else:
                                    if coloring[k] != coloring[first_touch[i][j]]:
                                        c = coloring[k]
                                        for l in range(mach.num_sites):
                                            if coloring[l] == c:
                                                coloring[l] = coloring[first_touch[i][j]]
            color_part = []
            for k in range(mach.num_sites):
                ins_set = []
                for i in range(len(mach.lines)):
                    line = mach.lines[i]
                    for j in range(len(line.inss)):
                        ins = line.inss[j]
                        if switch[evo_index][i][j] == 1 and coloring[first_touch[i][j]] == k:
                            ins_lvars = []
                            for l in range(len(ins.vars_index)):
                                ins_lvars.append(sol_lvars[ins.vars_index[l]])
                            ins_set.append(
                                (
                                    (i, j),
                                    ins,
                                    ins.exp_eval(sol_gvars, sol_lvars),
                                    ins_lvars,
                                )
                            )
                if ins_set:
                    color_part.append(ins_set)
            # Detach if commute with all others
            """
            If an instruction's Hamiltonian commutes with every
            other ones in the box, then we can set itself as a
            stand-alone box, detaching from the original box.
            """
            logger.info(color_part)
            for i in range(len(color_part)):
                ins_set = color_part[i]
                j = 0
                while j < len(ins_set):
                    all_commute = True
                    for k in range(len(ins_set)):
                        if j == k:
                            continue
                        # logger.info(j, k)
                        # logger.info(ins_set[j][2].ham, ins_set[k][2].ham)
                        if not TIHamiltonian.commutativity_test(
                            ins_set[j][2],
                            ins_set[k][2],
                            ins_set[j][1].nativeness == "derived"
                            or ins_set[k][1].nativeness == "derived",
                        ):
                            all_commute = False
                            break
                    if all_commute:
                        color_part.append([ins_set[j]])
                        del ins_set[j]
                        j -= 1
                    j += 1
            # Trotterization
            """
            The mutually exclusiveness can be modeled as a graph
            where vertices are instructions.

            An edge is added either both vertices share the same
            signal line, or they have shared sites and one of them
            is an derivied instruction.

            We use a coloring algorithm to minimize the number of
            colors, such that no edge has two vertices in the same
            color. With these coloring we can further divide a box
            into multiple parallel boxes, repeated multiply times
            (due to trotterization).
            """
            for ins_set in color_part:
                local_ending_boxes = ending_boxes
                n = len(ins_set)
                G = nx.Graph()
                G.add_nodes_from(range(n))
                for i in range(n):
                    for j in range(n):
                        if i == j:
                            continue
                        if (ins_set[i][0][0] == ins_set[j][0][0]) or (
                            (
                                ins_set[i][1].nativeness == "derived"
                                or ins_set[j][1].nativeness == "derived"
                            )
                            and (
                                not TIHamiltonian.commutativity_test(
                                    ins_set[i][2], ins_set[j][2], True
                                )
                            )
                        ):
                            G.add_edge(i, j)
                col = nx.coloring.greedy_color(G, strategy="largest_first")
                nodes_of_color = [[] for i in set(col.values())]
                for i in range(n):
                    nodes_of_color[col[i]].append(i)
                # Check if the partitions are pair-wise commute
                sumh = []
                for i in range(len(nodes_of_color)):
                    h = 0
                    for j in range(len(nodes_of_color[i])):
                        h = h + ins_set[nodes_of_color[i][j]][2]
                    sumh.append(h)
                all_commute = True
                for i in range(len(nodes_of_color)):
                    for j in range(i):
                        if not TIHamiltonian.commutativity_test(sumh[i], sumh[j]):
                            all_commute = False
                            break
                    if not all_commute:
                        break

                if all_commute:
                    order = 1
                    steps = 1
                    sequential = False
                else:
                    order = trotter_args["order"]
                    steps = trotter_args["num"] * (1 << (order - 1))
                    sequential = trotter_args["sequential"]

                for i in range(steps):
                    new_boxes = []
                    for k in range(len(nodes_of_color)):
                        list_ins = nodes_of_color[k]
                        new_boxes.append(len(boxes))
                        box_ins = []
                        for j in range(len(list_ins)):
                            box_ins.append(ins_set[list_ins[j]])
                        boxes.append((box_ins, sol_time[evo_index] / steps, sumh[k]))

                    if sequential:
                        if order == 1 or (order == 2 and i % 2 == 0):
                            for b in local_ending_boxes:
                                edges.append((b, new_boxes[0]))
                            for j in range(1, len(new_boxes)):
                                edges.append((new_boxes[j - 1], new_boxes[j]))
                            local_ending_boxes = [new_boxes[-1]]
                        elif order == 2 and i % 2 == 1:
                            for b in local_ending_boxes:
                                edges.append((b, new_boxes[-1]))
                            for j in range(len(new_boxes) - 1, 0, -1):
                                edges.append((new_boxes[j], new_boxes[j - 1]))
                            local_ending_boxes = [new_boxes[0]]
                    else:
                        for j in new_boxes:
                            for b in local_ending_boxes:
                                edges.append((b, j))
                        local_ending_boxes = new_boxes

                next_ending_boxes += local_ending_boxes

            ending_boxes = next_ending_boxes

        # Delete commutative edges
        """
        Clean up the result.
        Generate the minimal graph containing all relations.
        """
        G = nx.DiGraph()
        G.add_nodes_from(range(len(boxes)))
        s = 0
        while s < len(edges):
            edge = edges[s]
            if nx.has_path(G, edge[0], edge[1]):
                del edges[s]
                continue
            G.add_edge(*edge)
            s += 1
        s = 0
        while s < len(edges):
            edge = edges[s]
            G.remove_edge(*edge)
            if nx.has_path(G, edge[0], edge[1]):
                del edges[s]
                continue
            G.add_edge(*edge)
            h1 = boxes[edge[0]][2]
            h2 = boxes[edge[1]][2]
            if TIHamiltonian.commutativity_test(h1, h2):
                del edges[s]
                G.remove_edge(*edge)
                for i in range(len(edges)):
                    if edges[i][1] == edge[0]:
                        new_edge = (edges[i][0], edge[1])
                        if not nx.has_path(G, new_edge[0], new_edge[1]):
                            edges.append(new_edge)
                            G.add_edge(*new_edge)
                s -= 1
            s += 1

        logger.info(alignment)
        logger.info(sol_gvars)
        output_boxes = []
        for box in boxes:
            output_boxes.append(
                (
                    [((i, j), ins_lvars) for ((i, j), ins, h, ins_lvars) in box[0]],
                    box[1],
                )
            )
            logger.info(output_boxes[-1])
        for edge in edges:
            logger.info(edge)
        return alignment, sol_gvars, output_boxes, edges
    else:
        logger.info("No solution is found.")
        raise Exception("No solution is found.")
