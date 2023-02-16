from gurobipy import *


def resolve_1_2(Z, k):
    """
        Z List[int]: le vecteur d'utilités des agents
        k int
    """

    # Nombre d'agents
    n = len(Z)

    # Pour désactiver le output de gurobi
    env = Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    m = Model(f"D{k}", env=env)

    var = []

    # La variable rk
    var.append(m.addVar(vtype=GRB.CONTINUOUS, lb=-
                        float('inf'),  ub=float('inf'), name=f"r{k}"))

    # Les variables bik (<= 0)
    for i in range(n):
        var.append(m.addVar(vtype=GRB.CONTINUOUS, lb=-
                            float('inf'),  ub=0, name=f"b{i+1}_{k}"))

    m.update()

    # La fonction obj ; k rk + Somme(bik)
    obj = k * var[0]
    for i in range(n):
        obj += var[i+1]

    m.setObjective(obj, GRB.MAXIMIZE)

    # Les contraintes sous forme rk + bik <= zi
    for i in range(n):
        m.addConstr((var[0] + var[i+1]) <= Z[i], f"ctr {i+1}")

    m.optimize()
    return m.objVal


def resolve_1_4(coef, W):
    """
        coef : Matrice des coefficients des X dans Z
        W    : liste des poids
    """
    n, p = coef.shape

    # Calcul de W'
    for i in range(n-1):
        W[i] -= W[i+1]

    # Pour désactiver le output de gurobi
    env = Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    m = Model(f"Lin de f pour l'exemple 1", env=env)

    # Les n variables rk (R)
    var_r = []
    for k in range(n):
        var_r.append(m.addVar(vtype=GRB.CONTINUOUS, lb=-
                              float('inf'),  ub=float('inf'), name=f"r_{k+1}"))

    # Les p variables xi (binaire)
    var_x = []
    for i in range(p):
        var_x.append(m.addVar(vtype=GRB.BINARY, name=f"x_{i+1}"))

    # Les variables bik (<= 0)
    var_b = []
    for i in range(n):
        var_b.append([])
        for k in range(n):
            var_b[i].append(m.addVar(vtype=GRB.CONTINUOUS, lb=-
                                     float('inf'),  ub=0, name=f"b_{i+1}_{k+1}"))

    m.update()

    # Fonction Obj = somme(w'k * (k * rk + somme(bik)))
    obj = 0

    for k in range(n):
        temp = (k+1) * var_r[k]

        for i in range(n):
            temp += var_b[i][k]

        obj += W[k] * temp

    m.setObjective(obj, GRB.MAXIMIZE)

    # Les contraintes sous la forme rk + bik - zi <= 0
    # avec zi = coef[i] . X
    for k in range(n):
        for i in range(n):
            m.addConstr((var_r[k] + var_b[i][k] - quicksum(coef[i][x] * var_x[x]
                        for x in range(p))) <= 0, f"ctr_{k+1}_{i+1}")

    # Ajout de la contraintes du nombre d'objets choisis (somme xi = 3)
    m.addConstr((quicksum(var_x[x] for x in range(p)))
                == 3, f"ctr_x")

    try:
        m.optimize()

        print("La valeur objective est", m.objVal)

        for i in range(p):
            print(f"x{i+1} =", int(var_x[i].x), end=", ")

        rep = []
        for j in range(n):
            rep.append(0)
            for i in range(p):
                rep[j] += int(var_x[i].x * coef[j][i])

        rep = ", ".join(map(lambda x: str(x), rep))
        print(f"\n\nLes dotations sont de (Z=) : ({rep})")

    except Exception:
        print("Aucune solution")


def resolve_2_1(coef, W, verbose=True):
    """
        coef : Matrice des coefficients des X dans Z
        W    : liste des poids
    """

    n, p = coef.shape

    # Calcul de W'
    for i in range(n-1):
        W[i] -= W[i+1]

    # Pour désactiver le output de gurobi
    env = Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    m = Model("Lin f du partage équitable de biens", env=env)

    # Les variables rk (R)
    var_r = []
    for k in range(n):
        var_r.append(m.addVar(vtype=GRB.CONTINUOUS, lb=-
                              float('inf'),  ub=float('inf'), name=f"r_{k+1}"))

    # Les variables x (binaire)
    # avec xij vaut 1 si l'objet j est affecté à l'individu i, 0 sinon
    var_x = []
    for i in range(n):
        var_x.append([])
        for j in range(p):
            var_x[i].append(m.addVar(vtype=GRB.BINARY, name=f"x_{i+1}_{j+1}"))

    # Les variables bik (<= 0)
    var_b = []
    for i in range(n):
        var_b.append([])
        for k in range(n):
            var_b[i].append(m.addVar(vtype=GRB.CONTINUOUS, lb=-
                                     float('inf'),  ub=0, name=f"b_{i+1}_{k+1}"))

    m.update()

    # Fonction Obj = somme(w'k * (k * rk + somme(bik)))
    obj = 0
    for k in range(n):
        temp = (k+1) * var_r[k]

        for i in range(n):
            temp += var_b[i][k]

        obj += W[k] * temp

    m.setObjective(obj, GRB.MAXIMIZE)

    # Les contraintes sous la forme rk + bik - zi <= 0
    # avec zi = coef[i] . X
    for k in range(n):
        for i in range(n):
            m.addConstr((var_r[k] + var_b[i][k] - quicksum(coef[i][x] * var_x[i][x]
                        for x in range(p))) <= 0, f"ctr_{k+1}_{i+1}")

    # Contrainte que chaque objet est choisis au plus 1 fois
    for j in range(p):
        m.addConstr((quicksum(var_x[i][j]
                    for i in range(n)) <= 1), f"ctr_x_{j}")

    try:
        m.optimize()
        if verbose:
            print("La valeur objective du PL est", m.objVal, end="\n\n")

            print("          ", end="")
            for j in range(p):
                print(f"x{j+1}", end=" ")

            rep = []
            for i in range(n):
                rep.append(0)

                print(f"\nAgent_{i+1} :", end=" ")
                for j in range(p):
                    print(f" {int(var_x[i][j].x)}", end=" ")
                    rep[i] += int(var_x[i][j].x * coef[i][j])

            rep = ", ".join(map(lambda x: str(x), rep))
            print(f"\n\nLes dotations sont de (Z=) : ({rep})")

    except Exception:
        if verbose:
            print("Aucune solution")


def resolve_3_1(coef, W, C, b=None, verbose=True):
    """
        coef : Matrice des coefficients des X dans Z
        W    : liste des poids des zi
        C    : liste des coûts des X
        b    : l'enveloppe budgétaire max
    """

    if b is None:
        b = sum(C) / 2

    n, p = coef.shape

    # Calcul de W'
    for i in range(n-1):
        W[i] -= W[i+1]

    # Pour désactiver le output de gurobi
    env = Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    m = Model("Selection multicritère", env=env)

    # Les variables rk (R)
    var_r = []
    for k in range(n):
        var_r.append(m.addVar(vtype=GRB.CONTINUOUS, lb=-
                              float('inf'),  ub=float('inf'), name=f"r_{k+1}"))

    # Les variables xi (binaire)
    # vaut 1 si on choisis le projet i, 0 sinon
    var_x = []
    for i in range(p):
        var_x.append(m.addVar(vtype=GRB.BINARY, name=f"x_{i+1}"))

    # Les variables bik (<= 0)
    var_b = []
    for i in range(n):
        var_b.append([])
        for k in range(n):
            var_b[i].append(m.addVar(vtype=GRB.CONTINUOUS, lb=-
                                     float('inf'),  ub=0, name=f"b_{i+1}_{k+1}"))

    m.update()

    # Fonction Obj = somme(w'k * (k * rk + somme(bik)))
    obj = 0

    for k in range(n):
        temp = (k+1) * var_r[k]

        for i in range(n):
            temp += var_b[i][k]

        obj += W[k] * temp

    m.setObjective(obj, GRB.MAXIMIZE)

    # Les contraintes sous la forme rk + bik - zi <= 0
    # avec zi = coef[i] . X
    for k in range(n):
        for i in range(n):
            m.addConstr((var_r[k] + var_b[i][k] - quicksum(coef[i][x] * var_x[x]
                        for x in range(p))) <= 0, f"ctr_{k+1}_{i+1}")

    # Contrainte que le coût totale ne dépasse pas l'enveloppe budgétaire
    m.addConstr((quicksum(var_x[x] * C[x] for x in range(p)))
                <= b, "ctr_b")

    try:
        m.optimize()

        if verbose:
            print("La valeur objective est", m.objVal)

            for i in range(p):
                print(f"x{i+1} =", int(var_x[i].x), end=", ")

            rep = []
            for j in range(n):
                rep.append(0)
                for i in range(p):
                    rep[j] += int(var_x[i].x * coef[j][i])

            rep = ", ".join(map(lambda x: str(x), rep))
            print(f"\n\nLes dotations sont de (Z=) : ({rep})")

    except Exception:
        if verbose:
            print("Aucune solution")


def resolve_4_1(A, a, g):
    """
        A (n*n) avec n = nombre de noeuds
        matrice de transitions avec les poids des arcs pour un scénario donné
        a : noeud de départ
        g : noeud d'arrivé
    """

    n = len(A)
    d = [0] * n
    d[a] = 1
    d[g] = -1

    # Pour désactiver le output de gurobi
    env = Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    m = Model("Plus court chemin", env=env)

    arc_exis = []
    # Les variables xij (binaire)
    # xij vaut 1 si arc_ij appartient au chemin, 0 sinon
    var_x = []
    for i in range(n):
        for j in range(n):
            if A[i][j] != 0:
                var_x.append(
                    m.addVar(vtype=GRB.BINARY, name=f"x{i}_{j}"))

                arc_exis.append((i, j))
    m.update()

    # Fonction obj : somme(a_ij * x_ij)
    obj = 0

    for e, (i, j) in enumerate(arc_exis):
        obj += A[i][j] * var_x[e]

    m.setObjective(obj, GRB.MINIMIZE)

    # Les contraintes sous forme somme_j(xij) - somme_j(xji) == di
    for i in range(n):

        js = []
        je = []

        for e, (a, b) in enumerate(arc_exis):
            if a == i:
                js.append(e)
            if b == i:
                je.append(e)

        m.addConstr((quicksum(var_x[j] for j in js) -
                    quicksum(var_x[j] for j in je)) == d[i])

    try:
        m.optimize()

        print(f"\nLa valeur objective est {m.objVal}\n")

        for e, (i, j) in enumerate(arc_exis):
            if var_x[e].x != 0:
                print(f"x_{i}_{j}")

    except Exception:
        print("Aucune solution")


def resolve_4_2(A, W, a, g, res=False):
    """
        A (n*n) avec n = nombre de noeuds
        matrice de transitions avec les poids des arcs pour un scénario donné
        W : liste des poids
        a : noeud de départ
        g : noeud d'arrivé
    """

    A = -A
    n, p, _ = A.shape  # n nb scena, p nb points
    d = [0] * p
    d[a] = 1
    d[g] = -1

    # Calcul des W'
    for i in range(n-1):
        W[i] -= W[i+1]

    # Pour désactiver le output de gurobi
    env = Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    m = Model("Chemin robuste", env=env)

    arc_exis = []
    # Les variables xij (binaire)
    # xij vaut 1 si arc_ij appartient au chemin, 0 sinon
    var_x = []
    for i in range(p):
        for j in range(p):
            if A[0][i][j] != 0:
                var_x.append(
                    m.addVar(vtype=GRB.BINARY, name=f"x{i}_{j}"))

                arc_exis.append((i, j))

    # Les variables rk (R)
    var_r = []
    for k in range(n):
        var_r.append(m.addVar(vtype=GRB.CONTINUOUS, lb=-
                              float('inf'),  ub=float('inf'), name=f"r_{k+1}"))

    # Les variables bik (<= 0)
    var_b = []
    for i in range(n):
        var_b.append([])
        for k in range(n):
            var_b[i].append(m.addVar(vtype=GRB.CONTINUOUS, lb=-
                                     float('inf'),  ub=0, name=f"b_{i+1}_{k+1}"))

    m.update()

    # Fonction Obj = somme(w'k * (k * rk + somme(bik)))
    obj = 0

    for k in range(n):
        temp = (k+1) * var_r[k]

        for i in range(n):
            temp += var_b[i][k]

        obj += W[k] * temp

    m.setObjective(obj, GRB.MAXIMIZE)

    # Les contraintes sous forme rk + bik - somme(A_s_ij * xij)
    for k in range(n):
        for i in range(n):
            m.addConstr((var_r[k] + var_b[i][k] - quicksum(A[i][x][y] * var_x[e]
                        for e, (x, y) in enumerate(arc_exis))) <= 0, f"ctr_{k+1}_{i+1}")

    # Les contraintes sous forme somme_j(xij) - somme_j(xji) == di
    for i in range(p):

        js = []
        je = []

        # On extrait les listes des arcs sortants de i et entrants vers i
        for e, (a, b) in enumerate(arc_exis):
            if a == i:
                js.append(e)
            if b == i:
                je.append(e)

        # Somme des arcs sortant - Somme des arcs entrant = d[i]
        m.addConstr((quicksum(var_x[j] for j in js) -
                    quicksum(var_x[j] for j in je)) == d[i])

    try:
        m.optimize()

        if not res:
            print(f"\nLa valeur objective est {m.objVal}\n")

        ts = [0] * n

        for e, (i, j) in enumerate(arc_exis):
            if var_x[e].x != 0:
                if not res:
                    print(f"x_{i}_{j}")

                for k in range(n):
                    ts[k] += -A[k][i][j]

        if res:
            return ts

        ts = ", ".join(map(lambda x: str(x), ts))
        print(f"\n\nLes temps des scénarios est de : ({ts})")

    except Exception:
        if not res:
            print("Aucune solution")


def generate_w(alpha, n):
    """
        alpha: int
        n : le nombre d'agents
    """
    W = [0] * n

    for i in range(n):
        a = (n - (i+1) + 1) / n
        b = (n - (i+1)) / n

        W[i] = (a ** alpha) - (b ** alpha)

    return W
