/**
 * AlphaZero MCTS — port of giotto/agents/algorithms/alphazero/mcts.py
 *
 * Uses lazy env cloning and PUCT selection with neural network policy priors.
 * The `net` object must have a `predict(state)` method returning [policy, value].
 */

class AZNode {
    constructor(env = null, parent = null, parentAction = null, sourceEnv = null, sourceAction = null) {
        this.parent = parent;
        this.parentAction = parentAction;
        this._env = env;
        this._sourceEnv = sourceEnv;
        this._sourceAction = sourceAction;
        this.toPlay = env !== null ? env.currentPlayer : null;
        this.prob = 0.0;
        this.children = new Map();
        this.nVisits = 0;
        this.totalScore = 0.0;
    }

    get env() {
        if (this._env === null) {
            const env = this._sourceEnv.clone();
            env.step(this._sourceAction);
            this._env = env;
            this.toPlay = env.currentPlayer;
            this._sourceEnv = null;
            this._sourceAction = null;
        }
        return this._env;
    }

    get avgValue() {
        return this.nVisits === 0 ? 0.0 : this.totalScore / this.nVisits;
    }

    terminalEval(env, player) {
        const winner = env.info.winner;
        if (winner === -1) return 0.0;
        return winner === player ? 1.0 : -1.0;
    }

    expand() {
        if (this.children.size > 0 || this.env.done) return;
        for (const action of this.env.getValidActions()) {
            this.children.set(action, new AZNode(
                null, this, action, this._env, action
            ));
        }
    }

    selectChild(cpuct) {
        const sqrtParent = Math.sqrt(this.nVisits + 1);
        const cpuctSqrt = cpuct * sqrtParent;
        let bestChild = null;
        let bestScore = -1e9;
        for (const child of this.children.values()) {
            const score = -child.avgValue + cpuctSqrt * child.prob / (child.nVisits + 1);
            if (score > bestScore) {
                bestScore = score;
                bestChild = child;
            }
        }
        return bestChild;
    }

    backpropagate(value) {
        this.totalScore += value;
        this.nVisits++;
        if (this.parent !== null) {
            this.parent.backpropagate(-value);
        }
    }

    isLeaf() {
        return this.children.size === 0;
    }

    isTerminal() {
        return this.env.done;
    }
}

/**
 * Process board state into neural network input format.
 * Returns Float32Array of shape [1, 2, rows, cols].
 */
export function processState(state, rows, cols) {
    const [board, playerId] = state;
    const oppId = 1 - playerId;
    const size = rows * cols;
    const input = new Float32Array(2 * size);

    for (let i = 0; i < size; i++) {
        if (board[i] === playerId) input[i] = 1.0;
        if (board[i] === oppId) input[size + i] = 1.0;
    }
    return input;
}

/**
 * Softmax over an array of floats.
 */
function softmax(arr) {
    const max = Math.max(...arr);
    const exp = arr.map(x => Math.exp(x - max));
    const sum = exp.reduce((a, b) => a + b, 0);
    return exp.map(x => x / sum);
}

export class AlphaZeroMCTS {
    constructor(net, nSimulations, cpuct) {
        this.net = net;
        this.nSimulations = nSimulations;
        this.cpuct = cpuct;
        this._root = null;
    }

    _buildRoot(env) {
        const rootEnv = env.clone();
        const root = new AZNode(rootEnv);
        const validActions = rootEnv.getValidActions();
        const [policy] = this.net.predict(rootEnv.getState());

        const raw = new Float32Array(validActions.length);
        for (let i = 0; i < validActions.length; i++) {
            raw[i] = policy[validActions[i] - 1];
        }
        let sum = 0;
        for (let i = 0; i < raw.length; i++) sum += raw[i];
        for (let i = 0; i < raw.length; i++) raw[i] /= sum;

        for (let i = 0; i < validActions.length; i++) {
            const action = validActions[i];
            const child = new AZNode(null, root, action, rootEnv, action);
            child.prob = raw[i];
            root.children.set(action, child);
        }
        return root;
    }

    reset() {
        this._root = null;
    }

    run(env) {
        const root = this._root !== null ? this._root : this._buildRoot(env);

        for (let sim = 0; sim < this.nSimulations; sim++) {
            let node = root;

            // SELECTION
            while (!node.isTerminal() && !node.isLeaf()) {
                node = node.selectChild(this.cpuct);
            }

            // EXPANSION + EVALUATION
            let value;
            if (!node.isTerminal()) {
                node.expand();
                const [policy, val] = this.net.predict(node.env.getState());
                const validActions = node.env.getValidActions();
                const raw = new Float32Array(validActions.length);
                for (let i = 0; i < validActions.length; i++) {
                    raw[i] = policy[validActions[i] - 1];
                }
                let sum = 0;
                for (let i = 0; i < raw.length; i++) sum += raw[i];
                for (let i = 0; i < raw.length; i++) raw[i] /= sum;
                for (let i = 0; i < validActions.length; i++) {
                    node.children.get(validActions[i]).prob = raw[i];
                }
                value = val;
            } else {
                value = node.terminalEval(node.env, node.toPlay);
            }

            // BACKPROPAGATION
            node.backpropagate(value);
        }

        this._root = root;
        return this._selectAction(root);
    }

    _selectAction(root) {
        let bestAction = null;
        let bestVisits = -1;
        for (const [action, child] of root.children) {
            if (child.nVisits > bestVisits) {
                bestVisits = child.nVisits;
                bestAction = action;
            }
        }
        return bestAction;
    }
}
