#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <chrono>
#include <iomanip>
#include <omp.h>
//hola
// Estructura que representa una ciudad en un plano 2D
struct City {
    int id;
    double x;
    double y;
};

// Estructura que representa un individuo en la población (una ruta)
struct Individual {
    std::vector<int> route;
    double distance;
    double fitness;
};

// Generador de números aleatorios seguro para hilos (Thread-safe)
// Declarar thread_local asegura que cada hilo de OpenMP tenga su propia instancia del RNG.
std::mt19937& get_rng() {
    static thread_local std::mt19937 generator;
    static thread_local bool initialized = false;
    if (!initialized) {
        // Semilla combinando el tiempo del sistema, el ID del hilo y la entropía del hardware
        std::random_device rd;
        int tid = omp_get_thread_num();
        generator.seed(rd() ^ tid ^ (std::chrono::system_clock::now().time_since_epoch().count() + tid * 1000));
        initialized = true;
    }
    return generator;
}

// Utilidad para calcular la distancia euclidiana entre dos ciudades
double calculate_distance(const City& c1, const City& c2) {
    double dx = c1.x - c2.x;
    double dy = c1.y - c2.y;
    return std::sqrt(dx * dx + dy * dy);
}

// Calcular la matriz de distancias N x N una sola vez para optimizar la evaluación de aptitud (fitness)
void compute_distance_matrix(const std::vector<City>& cities, std::vector<std::vector<double>>& matrix) {
    int n = cities.size();
    matrix.resize(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            matrix[i][j] = calculate_distance(cities[i], cities[j]);
        }
    }
}

// Evaluar la distancia total de la ruta de un individuo
void evaluate_individual(Individual& ind, const std::vector<std::vector<double>>& dist_matrix) {
    double total_dist = 0.0;
    int n = ind.route.size();
    for (int i = 0; i < n - 1; ++i) {
        total_dist += dist_matrix[ind.route[i]][ind.route[i + 1]];
    }
    // Regresar a la ciudad de inicio para completar el viaje de ida y vuelta
    total_dist += dist_matrix[ind.route[n - 1]][ind.route[0]];
    ind.distance = total_dist;
    ind.fitness = 1.0 / (total_dist + 1e-6); // Evitar división por cero
}

// Generar un individuo con una ruta aleatoria
Individual create_random_individual(int num_cities, const std::vector<std::vector<double>>& dist_matrix) {
    Individual ind;
    ind.route.resize(num_cities);
    for (int i = 0; i < num_cities; ++i) {
        ind.route[i] = i;
    }
    // Mezclar la ruta usando el RNG local del hilo
    auto& rng = get_rng();
    std::shuffle(ind.route.begin(), ind.route.end(), rng);
    evaluate_individual(ind, dist_matrix);
    return ind;
}

// Operador de selección por torneo
Individual tournament_selection(const std::vector<Individual>& population, int tournament_size) {
    auto& rng = get_rng();
    std::uniform_int_distribution<int> dist(0, population.size() - 1);
    
    Individual best = population[dist(rng)];
    for (int i = 1; i < tournament_size; ++i) {
        const Individual& contender = population[dist(rng)];
        if (contender.fitness > best.fitness) {
            best = contender;
        }
    }
    return best;
}

// Operador de Cruce Ordenado (Ordered Crossover - OX)
// Preserva el orden relativo de los elementos de los padres para generar rutas válidas de TSP
void ordered_crossover(const Individual& p1, const Individual& p2, Individual& child1, Individual& child2, int num_cities) {
    child1.route.resize(num_cities);
    child2.route.resize(num_cities);
    
    auto& rng = get_rng();
    std::uniform_int_distribution<int> dist(0, num_cities - 1);
    
    int pt1 = dist(rng);
    int pt2 = dist(rng);
    if (pt1 > pt2) {
        std::swap(pt1, pt2);
    }
    
    if (pt1 == pt2) {
        child1 = p1;
        child2 = p2;
        return;
    }
    
    // Función lambda para generar un hijo a partir de dos padres
    auto breed_child = [&](const Individual& parent1, const Individual& parent2, Individual& child) {
        std::vector<bool> in_child(num_cities, false);
        
        // Copiar segmento desde parent1 al hijo
        for (int i = pt1; i <= pt2; ++i) {
            child.route[i] = parent1.route[i];
            in_child[parent1.route[i]] = true;
        }
        
        // Rellenar los espacios restantes en el hijo desde parent2, preservando el orden
        int target_idx = (pt2 + 1) % num_cities;
        int p2_idx = (pt2 + 1) % num_cities;
        
        for (int count = 0; count < num_cities; ++count) {
            int city = parent2.route[p2_idx];
            if (!in_child[city]) {
                child.route[target_idx] = city;
                target_idx = (target_idx + 1) % num_cities;
            }
            p2_idx = (p2_idx + 1) % num_cities;
        }
    };
    
    breed_child(p1, p2, child1);
    breed_child(p2, p1, child2);
}

// Operador de mutación por intercambio (Swap Mutation)
void mutate(Individual& ind, double mutation_rate, int num_cities) {
    auto& rng = get_rng();
    std::uniform_real_distribution<double> dist_prob(0.0, 1.0);
    
    if (dist_prob(rng) < mutation_rate) {
        std::uniform_int_distribution<int> dist_idx(0, num_cities - 1);
        int idx1 = dist_idx(rng);
        int idx2 = dist_idx(rng);
        std::swap(ind.route[idx1], ind.route[idx2]);
    }
}

// Utilidad para encontrar el mejor individuo en una población
Individual find_best_individual(const std::vector<Individual>& population) {
    Individual best = population[0];
    for (size_t i = 1; i < population.size(); ++i) {
        if (population[i].distance < best.distance) {
            best = population[i];
        }
    }
    return best;
}

// -------------------------------------------------------------
// Algoritmo Genético Secuencial
// -------------------------------------------------------------
Individual run_sequential_ga(
    std::vector<Individual> population, // Se pasa por valor para trabajar sobre una copia
    const std::vector<std::vector<double>>& dist_matrix,
    int num_cities,
    int pop_size,
    int generations,
    double mutation_rate,
    int tournament_size,
    double& execution_time
) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    Individual best_overall = find_best_individual(population);
    std::vector<Individual> next_generation(pop_size);
    
    for (int gen = 1; gen <= generations; ++gen) {
        
        // Crear la siguiente generación
        for (int i = 0; i < pop_size; i += 2) {
            Individual parent1 = tournament_selection(population, tournament_size);
            Individual parent2 = tournament_selection(population, tournament_size);
            
            Individual child1, child2;
            ordered_crossover(parent1, parent2, child1, child2, num_cities);
            
            mutate(child1, mutation_rate, num_cities);
            mutate(child2, mutation_rate, num_cities);
            
            evaluate_individual(child1, dist_matrix);
            evaluate_individual(child2, dist_matrix);
            
            next_generation[i] = child1;
            if (i + 1 < pop_size) {
                next_generation[i + 1] = child2;
            }
        }
        
        // Elitismo: Preservar el mejor individuo de la generación anterior
        Individual best_in_gen = find_best_individual(population);
        if (best_in_gen.distance < best_overall.distance) {
            best_overall = best_in_gen;
        }
        // Reemplazar al peor individuo de la nueva generación con el mejor global
        int worst_idx = 0;
        for (int i = 1; i < pop_size; ++i) {
            if (next_generation[i].distance > next_generation[worst_idx].distance) {
                worst_idx = i;
            }
        }
        next_generation[worst_idx] = best_overall;
        
        // Intercambiar poblaciones
        population = next_generation;
        
        if (gen % 50 == 0 || gen == generations) {
            std::cout << "[Secuencial] Gen " << std::setw(3) << gen 
                      << " | Dist. Mínima: " << std::fixed << std::setprecision(2) << best_overall.distance << std::endl;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    execution_time = std::chrono::duration<double>(end_time - start_time).count();
    
    return best_overall;
}

// -------------------------------------------------------------
// Algoritmo Genético en Paralelo usando OpenMP
// -------------------------------------------------------------
Individual run_parallel_ga(
    std::vector<Individual> population, // Se pasa por valor para trabajar sobre una copia
    const std::vector<std::vector<double>>& dist_matrix,
    int num_cities,
    int pop_size,
    int generations,
    double mutation_rate,
    int tournament_size,
    double& execution_time
) {
    double start_time = omp_get_wtime();
    
    Individual best_overall = find_best_individual(population);
    std::vector<Individual> next_generation(pop_size);
    
    for (int gen = 1; gen <= generations; ++gen) {
        
        // Bucle paralelo para selección, cruce, mutación y evaluación de hijos
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < pop_size; i += 2) {
            // Selección por torneo (solo lectura, segura para hilos)
            Individual parent1 = tournament_selection(population, tournament_size);
            Individual parent2 = tournament_selection(population, tournament_size);
            
            // Cruce Ordenado (OX)
            Individual child1, child2;
            ordered_crossover(parent1, parent2, child1, child2, num_cities);
            
            // Mutación
            mutate(child1, mutation_rate, num_cities);
            mutate(child2, mutation_rate, num_cities);
            
            // Evaluación de aptitud
            evaluate_individual(child1, dist_matrix);
            evaluate_individual(child2, dist_matrix);
            
            next_generation[i] = child1;
            if (i + 1 < pop_size) {
                next_generation[i + 1] = child2;
            }
        }
        
        // Buscar el mejor individuo de la generación actual en paralelo
        Individual best_in_gen = population[0];
        #pragma omp parallel
        {
            Individual local_best = population[0];
            #pragma omp for nowait
            for (int i = 1; i < pop_size; ++i) {
                if (population[i].distance < local_best.distance) {
                    local_best = population[i];
                }
            }
            #pragma omp critical
            {
                if (local_best.distance < best_in_gen.distance) {
                    best_in_gen = local_best;
                }
            }
        }
        
        if (best_in_gen.distance < best_overall.distance) {
            best_overall = best_in_gen;
        }
        
        // Elitismo: Buscar al peor de next_generation para reemplazarlo con el mejor global
        // Se realiza secuencialmente ya que es una única operación muy rápida
        int worst_idx = 0;
        for (int i = 1; i < pop_size; ++i) {
            if (next_generation[i].distance > next_generation[worst_idx].distance) {
                worst_idx = i;
            }
        }
        next_generation[worst_idx] = best_overall;
        
        // Intercambiar poblaciones
        population = next_generation;
        
        if (gen % 50 == 0 || gen == generations) {
            #pragma omp single
            {
                std::cout << "[Paralelo]   Gen " << std::setw(3) << gen 
                          << " | Dist. Mínima: " << std::fixed << std::setprecision(2) << best_overall.distance << std::endl;
            }
        }
    }
    
    double end_time = omp_get_wtime();
    execution_time = end_time - start_time;
    
    return best_overall;
}

int main() {
    // Parámetros del problema
    const int num_cities = 150;        // Número de ciudades del problema TSP
    const int pop_size = 8000;         // Tamaño de la población del algoritmo genético (debe ser par)
    const int generations = 200;       // Número de generaciones
    const double mutation_rate = 0.08; // Probabilidad de mutación
    const int tournament_size = 5;     // Tamaño del torneo para selección
    const int width = 1000;            // Límites para las coordenadas X de las ciudades
    const int height = 1000;           // Límites para las coordenadas Y de las ciudades

    std::cout << "==========================================================" << std::endl;
    std::cout << "     ALGORITMO GENETICO EN PARALELO PARA TSP CON OpenMP   " << std::endl;
    std::cout << "==========================================================" << std::endl;
    std::cout << "Información del Sistema:" << std::endl;
    
    #pragma omp parallel
    {
        #pragma omp single
        {
            std::cout << "  Hilos máximos de OpenMP disponibles: " << omp_get_max_threads() << std::endl;
            std::cout << "  Hilos activos ejecutándose:          " << omp_get_num_threads() << std::endl;
        }
    }
    
    std::cout << "\nParámetros del Algoritmo Genético:" << std::endl;
    std::cout << "  Ciudades:            " << num_cities << std::endl;
    std::cout << "  Población:           " << pop_size << std::endl;
    std::cout << "  Generaciones:        " << generations << std::endl;
    std::cout << "  Tasa de Mutación:    " << mutation_rate * 100.0 << "%" << std::endl;
    std::cout << "  Tamaño del Torneo:   " << tournament_size << std::endl;
    std::cout << "==========================================================\n" << std::endl;

    // 1. Inicializar las ciudades de forma aleatoria (con semilla fija para reproducibilidad)
    std::vector<City> cities(num_cities);
    std::mt19937 init_rng(42);
    std::uniform_real_distribution<double> dist_x(0.0, width);
    std::uniform_real_distribution<double> dist_y(0.0, height);
    
    for (int i = 0; i < num_cities; ++i) {
        cities[i] = {i, dist_x(init_rng), dist_y(init_rng)};
    }

    // 2. Pre-calcular la matriz de distancias
    std::vector<std::vector<double>> distance_matrix;
    compute_distance_matrix(cities, distance_matrix);

    // 3. Crear población inicial idéntica para ambas ejecuciones
    std::vector<Individual> initial_population(pop_size);
    std::mt19937 pop_rng(1337);
    
    // Generación secuencial de la población inicial aleatoria
    for (int i = 0; i < pop_size; ++i) {
        Individual ind;
        ind.route.resize(num_cities);
        for (int j = 0; j < num_cities; ++j) {
            ind.route[j] = j;
        }
        std::shuffle(ind.route.begin(), ind.route.end(), pop_rng);
        evaluate_individual(ind, distance_matrix);
        initial_population[i] = ind;
    }

    std::cout << "Distancia inicial de la mejor ruta: " 
              << find_best_individual(initial_population).distance << "\n" << std::endl;

    // 4. Ejecutar GA Secuencial
    double serial_time = 0.0;
    std::cout << "Ejecutando Algoritmo Genético Secuencial..." << std::endl;
    Individual best_serial = run_sequential_ga(
        initial_population, distance_matrix, num_cities, pop_size, 
        generations, mutation_rate, tournament_size, serial_time
    );
    std::cout << "Ejecución Secuencial Finalizada en " << serial_time << " segundos.\n" << std::endl;

    // 5. Ejecutar GA en Paralelo (OpenMP)
    double parallel_time = 0.0;
    std::cout << "Ejecutando Algoritmo Genético en Paralelo (OpenMP)..." << std::endl;
    Individual best_parallel = run_parallel_ga(
        initial_population, distance_matrix, num_cities, pop_size, 
        generations, mutation_rate, tournament_size, parallel_time
    );
    std::cout << "Ejecución en Paralelo Finalizada en " << parallel_time << " segundos.\n" << std::endl;

    // 6. Reportar los resultados del Benchmark y Aceleración
    double speedup = serial_time / parallel_time;
    double efficiency = (speedup / omp_get_max_threads()) * 100.0;

    std::cout << "==========================================================" << std::endl;
    std::cout << "                   RESUMEN DE BENCHMARK                   " << std::endl;
    std::cout << "==========================================================" << std::endl;
    std::cout << "  Tiempo Secuencial:     " << std::fixed << std::setprecision(4) << serial_time << " s" << std::endl;
    std::cout << "  Tiempo en Paralelo:    " << std::fixed << std::setprecision(4) << parallel_time << " s" << std::endl;
    std::cout << "  Aceleración (Speedup): " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
    std::cout << "  Eficiencia:            " << std::fixed << std::setprecision(2) << efficiency << "%" << std::endl;
    std::cout << "----------------------------------------------------------" << std::endl;
    std::cout << "  Mejor Ruta Secuencial:   " << best_serial.distance << std::endl;
    std::cout << "  Mejor Ruta en Paralelo:  " << best_parallel.distance << std::endl;
    std::cout << "==========================================================" << std::endl;

    return 0;
}
