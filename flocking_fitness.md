Implement the fitness function exactly following the paper "Evolving flocking in embodied agents based on local and global application of Reynolds’ rules" (Ramos et al., 2019).

IMPORTANT:
Do NOT implement the Local setup. Implement ONLY the successful Global setup (Eq.11).

The fitness of one episode is:

Fg = (1/T) * Σ_t [ Cg(t) + S(t) + Ag(t) ] + M

where:

1. Global Alignment Ag(t)

Ag(t) = | (1/N) * Σ_i exp(j * theta_i) |

Implementation:

```python
vx = np.cos(theta)
vy = np.sin(theta)

Ag = np.sqrt(vx.mean()**2 + vy.mean()**2)
```

Range: [0,1]

Ag=1 means all agents have identical heading.

2. Global Cohesion Cg(t)

The swarm is treated as an undirected graph.

Two agents belong to the same group if:

distance(i,j) < neighbor_radius

Compute the number of connected components:

num_groups

Then:

Cg = 1.0 / num_groups

Examples:

1 group -> Cg=1

2 groups -> Cg=0.5

4 groups -> Cg=0.25

3. Separation S(t)

Count collisions:

collision_i = 1 if agent i collides with any other agent
collision_i = 0 otherwise

Then:

S = 1 - collisions.mean()

Thus:

No collisions -> S=1

All robots collide -> S=0

4. Movement bonus M

Let:

d = average displacement of all agents from their initial positions

D = target displacement (paper uses D=5m)

Then:

M = min(d / D, 1)

Range:

0 <= M <= 1

5. Final implementation

Pseudo-code:

```python
fitness_sum = 0

for t in range(T):

    Ag = global_alignment(theta)

    Cg = 1 / number_of_connected_groups(pos)

    S = 1 - collision_ratio(pos)

    fitness_sum += Ag + Cg + S

fitness = fitness_sum / T

fitness += movement_bonus(initial_pos, final_pos)
```

Important:

* Cohesion MUST be GLOBAL, not neighbor average.
* Alignment MUST be GLOBAL order parameter.
* Groups MUST be computed as connected components.
* This is the key finding of the paper: Global fitness evolves a single flock, while Local fitness only evolves fragmented local flocks.
