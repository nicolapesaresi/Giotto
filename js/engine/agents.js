/**
 * Simple agents — port of giotto/agents/random.py and minimax.py
 */

export class RandomAgent {
    constructor() {
        this.name = "Random";
    }

    selectAction(env) {
        const actions = env.getValidActions();
        return actions[Math.floor(Math.random() * actions.length)];
    }
}

export class MinimaxAgent {
    constructor() {
        this.name = "Minimax";
        this.playerId = null;
    }

    selectAction(env) {
        if (this.playerId === null) this.playerId = env.currentPlayer;

        let bestScore = -Infinity;
        let bestAction = null;

        for (const action of env.getValidActions()) {
            const sim = env.clone();
            sim.step(action);
            const score = this._minimax(sim, 1);
            if (score > bestScore) {
                bestScore = score;
                bestAction = action;
            }
        }
        return bestAction;
    }

    _minimax(env, depth) {
        if (env.done) return this._evaluate(env, depth);

        if (env.currentPlayer === this.playerId) {
            let best = -Infinity;
            for (const action of env.getValidActions()) {
                const sim = env.clone();
                sim.step(action);
                best = Math.max(best, this._minimax(sim, depth + 1));
            }
            return best;
        } else {
            let best = Infinity;
            for (const action of env.getValidActions()) {
                const sim = env.clone();
                sim.step(action);
                best = Math.min(best, this._minimax(sim, depth + 1));
            }
            return best;
        }
    }

    _evaluate(env, depth) {
        const winner = env.info.winner;
        if (winner === -1) return 0;
        if (winner === this.playerId) return 10 - depth;
        return depth - 10;
    }
}
