import java.util.*;

/**
 * Lab 08, Project 2: Q-Learning maze solver.
 * Actions: 0=up, 1=right, 2=down, 3=left.
 */
public class QLearningMazeSolver {
    static final int ROWS = 5, COLS = 5;
    static final int START = 0, GOAL = 24;
    static final double ALPHA = 0.15, GAMMA = 0.95;
    static final int EPISODES = 5000, MAX_STEPS = 100;
    static final double EPSILON_START = 1.0, EPSILON_MIN = 0.03, EPSILON_DECAY = 0.995;
    // 1 means blocked wall; S=(0,0), G=(4,4)
    static final boolean[] WALL = new boolean[ROWS * COLS];
    static final int[] DR = {-1, 0, 1, 0}, DC = {0, 1, 0, -1};
    static double[][] q = new double[ROWS * COLS][4];
    static Random random = new Random(42);

    static {
        int[] walls = {2, 7, 12, 17, 20}; // fixed maze obstacles
        for (int w : walls) WALL[w] = true;
        WALL[START] = false;
        WALL[GOAL] = false;
    }

    static int move(int state, int action) {
        int r = state / COLS, c = state % COLS;
        int nr = r + DR[action], nc = c + DC[action];
        if (nr < 0 || nr >= ROWS || nc < 0 || nc >= COLS) return state;
        int next = nr * COLS + nc;
        return WALL[next] ? state : next;
    }

    static int chooseAction(int state, double epsilon) {
        if (random.nextDouble() < epsilon) return random.nextInt(4);
        int best = 0;
        for (int a = 1; a < 4; a++) if (q[state][a] > q[state][best]) best = a;
        return best;
    }

    public static void main(String[] args) {
        double epsilon = EPSILON_START;
        for (int ep = 1; ep <= EPISODES; ep++) {
            int state = START;
            double total = 0;
            for (int step = 0; step < MAX_STEPS; step++) {
                int action = chooseAction(state, epsilon);
                int next = move(state, action);
                boolean done = next == GOAL;
                double reward = done ? 100 : (next == state ? -5 : -1);
                double maxNext = Arrays.stream(q[next]).max().orElse(0);
                q[state][action] += ALPHA * (reward + (done ? 0 : GAMMA * maxNext) - q[state][action]);
                state = next;
                total += reward;
                if (done) break;
            }
            epsilon = Math.max(EPSILON_MIN, epsilon * EPSILON_DECAY);
            if (ep % 500 == 0) System.out.printf("Episode %4d | epsilon %.3f | reward %.1f%n", ep, epsilon, total);
        }

        System.out.println("\\nMaze (* = learned route, # = wall, S = start, G = goal):");
        printRoute();
    }

    static void printRoute() {
        int state = START;
        boolean[] route = new boolean[ROWS * COLS];
        route[state] = true;
        Set<Integer> seen = new HashSet<>();
        seen.add(state);
        for (int i = 0; i < ROWS * COLS; i++) {
            if (state == GOAL) break;
            int action = chooseGreedy(state);
            int next = move(state, action);
            if (next == state || seen.contains(next)) {
                // Pick a legal unvisited neighbor with the highest Q-value.
                double best = -Double.MAX_VALUE; int candidate = state;
                for (int a = 0; a < 4; a++) {
                    int n = move(state, a);
                    if (n != state && !seen.contains(n) && q[state][a] > best) {
                        best = q[state][a]; candidate = n;
                    }
                }
                next = candidate;
            }
            if (next == state) break;
            state = next; route[state] = true; seen.add(state);
        }
        for (int r = 0; r < ROWS; r++) {
            for (int c = 0; c < COLS; c++) {
                int s = r * COLS + c;
                char ch = s == START ? 'S' : s == GOAL ? 'G' : WALL[s] ? '#' : route[s] ? '*' : '.';
                System.out.print(ch + " ");
            }
            System.out.println();
        }
        if (route[GOAL]) System.out.println("Goal reached by learned policy.");
        else System.out.println("Route extraction stopped before the goal; rerun or tune episodes/parameters.");
    }

    static int chooseGreedy(int state) {
        int best = 0;
        for (int a = 1; a < 4; a++) if (q[state][a] > q[state][best]) best = a;
        return best;
    }
}
