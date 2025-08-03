import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import os


def cargar_instancia_vrp(ruta_archivo):
    with open(ruta_archivo, 'r') as f:
        lineas = [l.strip() for l in f.readlines()]
    dimension = vehiculos = None
    deposito = None
    matriz = []
    leyendo_matriz = False
    leyendo_depot = False

    for linea in lineas:
        if linea.startswith("DIMENSION"):
            dimension = int(linea.split(":")[-1])
        elif linea.startswith("VEHICLES"):
            vehiculos = int(linea.split(":")[-1])
        elif linea.startswith("EDGE_WEIGHT_SECTION"):
            leyendo_matriz = True
            continue
        elif linea.startswith("DEMAND_SECTION"):
            leyendo_matriz = False
            continue
        elif linea.startswith("DEPOT_SECTION"):
            leyendo_depot = True
            continue
        elif linea.startswith("EOF"):
            leyendo_matriz = False
            leyendo_depot = False
            continue
        if leyendo_matriz and linea:
            matriz.extend(map(int, linea.split()))
        elif leyendo_depot and linea.isdigit():
            if deposito is None:
                deposito = int(linea) - 1 
    matriz_distancias = np.array(matriz).reshape((dimension, dimension))
    return dimension, vehiculos, deposito, matriz_distancias



def calcular_ruta_distancia(ruta_1based, matriz, depot_idx0):
    if not ruta_1based:
        return 0
    idx_prim = ruta_1based[0] - 1
    dist = matriz[depot_idx0, idx_prim]
    for i in range(len(ruta_1based) - 1):
        a = ruta_1based[i] - 1
        b = ruta_1based[i+1] - 1
        dist += matriz[a, b]
    idx_ult = ruta_1based[-1] - 1
    dist += matriz[idx_ult, depot_idx0]
    return dist



def aplicar_2opt(ruta_1based, matriz, depot_idx0):
    improved = True
    while improved:
        improved = False
        for i in range(1, len(ruta_1based) - 2):
            for j in range(i + 1, len(ruta_1based)):
                if j - i == 1:
                    continue
                nueva = ruta_1based[:i] + ruta_1based[i:j][::-1] + ruta_1based[j:]
                if calcular_ruta_distancia(nueva, matriz, depot_idx0) < \
                   calcular_ruta_distancia(ruta_1based, matriz, depot_idx0):
                    ruta_1based = nueva
                    improved = True
    return ruta_1based



def fitness(sol, matriz, n_ciudades, depot_idx0, peso_deseq=0):
    visitadas = set()
    penal = 0
    distancias = []
    for ruta in sol:
        for c in ruta:
            if c in visitadas:
                penal += 10000
            visitadas.add(c)
        distancias.append(calcular_ruta_distancia(ruta, matriz, depot_idx0))

    faltantes = (n_ciudades - 1) - len(visitadas)
    if faltantes > 0:
        penal += 10000 * faltantes

    desequilibrio = np.std(distancias) if len(distancias) > 1 else 0
    return sum(distancias) + penal + peso_deseq * desequilibrio



def crear_individuo(ciudades_1based, vehs):
    random.shuffle(ciudades_1based)
    rutas = [[] for _ in range(vehs)]
    for i, c in enumerate(ciudades_1based):
        rutas[i % vehs].append(c)
    return rutas



def cruce_balanceado(p1, p2):
    n = max(len(p1), len(p2))
    hijo = [[] for _ in range(n)]
    usados = set()
    for padre in (p1, p2):
        for i, ruta in enumerate(padre):
            for c in ruta:
                if c not in usados:
                    hijo[i % n].append(c)
                    usados.add(c)
    return hijo



def mutacion_move(sol):
    if len(sol) < 2:
        return
    r1, r2 = random.sample(range(len(sol)), 2)
    if sol[r1]:
        i = random.randint(0, len(sol[r1]) - 1)
        ciudad = sol[r1].pop(i)
        sol[r2].insert(random.randint(0, len(sol[r2])), ciudad)



def torneo(poblacion, matriz, n_ciudades, depot_idx0, k=3):
    elegidos = random.sample(poblacion, k)
    return min(elegidos, key=lambda ind: fitness(ind, matriz, n_ciudades, depot_idx0))



def algoritmo_genetico(ciudades_1based, matriz, vehs, depot_idx0, gens=300, tam=100):
    poblacion = [crear_individuo(ciudades_1based[:], vehs) for _ in range(tam)]
    mejor = min(poblacion, key=lambda ind: fitness(ind, matriz, len(ciudades_1based)+1, depot_idx0))
    mejor_score = fitness(mejor, matriz, len(ciudades_1based)+1, depot_idx0)
    historial = [mejor_score]

    for _ in tqdm(range(gens), desc="Progreso"):
        nueva = [mejor]
        while len(nueva) < tam:
            p1 = torneo(poblacion, matriz, len(ciudades_1based)+1, depot_idx0)
            p2 = torneo(poblacion, matriz, len(ciudades_1based)+1, depot_idx0)
            hijo = cruce_balanceado(p1, p2)
            if random.random() < 0.3:
                mutacion_move(hijo)
            hijo = [aplicar_2opt(r, matriz, depot_idx0) for r in hijo]
            nueva.append(hijo)
        poblacion = nueva
        candidato = min(poblacion, key=lambda ind: fitness(ind, matriz, len(ciudades_1based)+1, depot_idx0))
        score = fitness(candidato, matriz, len(ciudades_1based)+1, depot_idx0)
        if score < mejor_score:
            mejor, mejor_score = candidato, score
        historial.append(mejor_score)

    return mejor, mejor_score, historial



def exportar(nombre, mejor_sol, mejor_coste, historial, matriz, depot_idx0):
    base = "final"
    carpeta = os.path.join(base, os.path.splitext(nombre)[0])
    os.makedirs(carpeta, exist_ok=True)
    with open(os.path.join(carpeta, "mejor_solucion.txt"), "w", encoding="utf-8") as f:
        f.write("MEJOR SOLUCIÓN\n")
        f.write(f"→ Costo total: {mejor_coste:.0f}\n")
        f.write(f"→ Vehículos usados: {len(mejor_sol)}\n")
        f.write(f"→ Ciudades cubiertas: {sum(len(r) for r in mejor_sol)}\n\n")
        for i, ruta_1based in enumerate(mejor_sol):
            ruta_str = " → ".join(str(c) for c in ruta_1based)
            f.write(f"Vehículo {i+1}: {depot_idx0+1} → {ruta_str} → {depot_idx0+1}\n")

    df = pd.DataFrame({
        "Vehículo": [i+1 for i in range(len(mejor_sol))],
        "Nº ciudades": [len(r) for r in mejor_sol],
        "Costo individual": [
            calcular_ruta_distancia(r, matriz, depot_idx0) for r in mejor_sol
        ]
    })
    df.to_csv(os.path.join(carpeta, "estadisticas.csv"), index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(historial, color='green', marker='o')
    plt.title("Costo por generación")
    plt.xlabel("Generación")
    plt.ylabel("Costo")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(carpeta, "mejor_costos.png"))
    plt.close()

    angulos = np.linspace(0, 2 * np.pi, matriz.shape[0], endpoint=False)
    coords = {i: (np.cos(a), np.sin(a)) for i, a in enumerate(angulos)}
    coords[depot_idx0] = (0, 0)  
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor('#f0f0f0')
    colores = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for i, ruta_1based in enumerate(mejor_sol):
        puntos = [depot_idx0] + [c-1 for c in ruta_1based] + [depot_idx0]
        x = [coords[p][0] for p in puntos]
        y = [coords[p][1] for p in puntos]
        ax.plot(x, y, label=f"Vehículo {i+1}", marker='o', color=colores[i % len(colores)])
        for c in ruta_1based:
            idx0 = c - 1
            ax.text(coords[idx0][0], coords[idx0][1], str(c), fontsize=8, ha='center')
    ax.scatter(coords[depot_idx0][0], coords[depot_idx0][1],
               c='red', s=200, marker='*', label=f"Depósito ({depot_idx0+1})")
    ax.set_title("Rutas encontradas")
    ax.axis('off')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(carpeta, "rutas.png"))
    plt.close()



if __name__ == "__main__":
    archivo = "A045-03f.dat"
    dim, vehs, depot_idx0, matriz = cargar_instancia_vrp(archivo)
    ciudades_1based = list(range(1, dim + 1))
    ciudades_1based.remove(depot_idx0 + 1)

    mejor_sol, mejor_coste, historial = algoritmo_genetico(
        ciudades_1based, matriz, vehs, depot_idx0
    )
    exportar(archivo, mejor_sol, mejor_coste, historial, matriz, depot_idx0)
