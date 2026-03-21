/**
 * Web Worker for AI computation.
 * Self-contained: includes env/MCTS logic to avoid module worker compatibility issues.
 * Uses importScripts for ONNX Runtime Web.
 */

// ============================================================
// ONNX Runtime
// ============================================================
importScripts("https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.1/dist/ort.min.js");
ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.1/dist/";

// ============================================================
// Environments (inlined from engine/)
// ============================================================

class TrisEnv {
    constructor() {
        this.rows = 3; this.cols = 3;
        this.board = new Int8Array(9);
        this.reset(1);
    }
    reset(sp) {
        this.currentPlayer = sp;
        this.board.fill(-1);
        this.turnCounter = 0;
        this.done = false;
        this.info = { moves: [] };
    }
    step(action) {
        this.info.moves.push(action);
        const idx = action - 1;
        this.board[idx] = this.currentPlayer;
        this.turnCounter++;
        if (this.checkWin(this.currentPlayer)) {
            this.done = true; this.info.winner = this.currentPlayer;
        } else if (this.turnCounter === 9) {
            this.done = true; this.info.winner = -1;
        }
        this.currentPlayer = 1 - this.currentPlayer;
    }
    getValidActions() {
        const a = [];
        for (let i = 0; i < 9; i++) { if (this.board[i] === -1) a.push(i + 1); }
        return a;
    }
    getState() { return [Int8Array.from(this.board), this.currentPlayer]; }
    checkWin(p) {
        const b = this.board;
        for (let r = 0; r < 3; r++) { const i = r * 3; if (b[i]===p && b[i+1]===p && b[i+2]===p) return true; }
        for (let c = 0; c < 3; c++) { if (b[c]===p && b[c+3]===p && b[c+6]===p) return true; }
        if (b[0]===p && b[4]===p && b[8]===p) return true;
        if (b[2]===p && b[4]===p && b[6]===p) return true;
        return false;
    }
    clone() {
        const e = Object.create(TrisEnv.prototype);
        e.rows = 3; e.cols = 3;
        e.board = Int8Array.from(this.board);
        e.currentPlayer = this.currentPlayer;
        e.turnCounter = this.turnCounter;
        e.done = this.done;
        e.info = { moves: [...this.info.moves] };
        if ("winner" in this.info) e.info.winner = this.info.winner;
        return e;
    }
}

class Connect4Env {
    constructor() {
        this.rows = 6; this.cols = 7;
        this.board = new Int8Array(42);
        this.reset(1);
    }
    reset(sp) {
        this.currentPlayer = sp;
        this.board.fill(-1);
        this.turnCounter = 0;
        this.done = false;
        this.info = { moves: [] };
    }
    step(action) {
        this.info.moves.push(action);
        const col = action - 1;
        let row = -1;
        for (let r = 0; r < 6; r++) { if (this.board[r * 7 + col] === -1) { row = r; break; } }
        this.board[row * 7 + col] = this.currentPlayer;
        this.turnCounter++;
        if (this.checkWin(this.currentPlayer)) {
            this.done = true; this.info.winner = this.currentPlayer;
        } else if (this.turnCounter === 42) {
            this.done = true; this.info.winner = -1;
        }
        this.currentPlayer = 1 - this.currentPlayer;
    }
    getValidActions() {
        const a = [];
        for (let c = 0; c < 7; c++) {
            for (let r = 0; r < 6; r++) {
                if (this.board[r * 7 + c] === -1) { a.push(c + 1); break; }
            }
        }
        return a;
    }
    getState() { return [Int8Array.from(this.board), this.currentPlayer]; }
    checkWin(p) {
        const b = this.board;
        for (let r = 0; r < 6; r++) for (let c = 0; c <= 3; c++) {
            const i = r*7+c;
            if (b[i]===p && b[i+1]===p && b[i+2]===p && b[i+3]===p) return true;
        }
        for (let r = 0; r <= 2; r++) for (let c = 0; c < 7; c++) {
            const i = r*7+c;
            if (b[i]===p && b[i+7]===p && b[i+14]===p && b[i+21]===p) return true;
        }
        for (let r = 0; r <= 2; r++) for (let c = 0; c <= 3; c++) {
            const i = r*7+c;
            if (b[i]===p && b[i+8]===p && b[i+16]===p && b[i+24]===p) return true;
        }
        for (let r = 0; r <= 2; r++) for (let c = 3; c < 7; c++) {
            const i = r*7+c;
            if (b[i]===p && b[i+6]===p && b[i+12]===p && b[i+18]===p) return true;
        }
        return false;
    }
    clone() {
        const e = Object.create(Connect4Env.prototype);
        e.rows = 6; e.cols = 7;
        e.board = Int8Array.from(this.board);
        e.currentPlayer = this.currentPlayer;
        e.turnCounter = this.turnCounter;
        e.done = this.done;
        e.info = { moves: [...this.info.moves] };
        if ("winner" in this.info) e.info.winner = this.info.winner;
        return e;
    }
}

// ============================================================
// Simple agents
// ============================================================

function randomAction(env) {
    const actions = env.getValidActions();
    return actions[Math.floor(Math.random() * actions.length)];
}

function minimaxAction(env) {
    const playerId = env.currentPlayer;
    let bestScore = -Infinity, bestAction = null;

    for (const action of env.getValidActions()) {
        const sim = env.clone(); sim.step(action);
        const score = minimax(sim, 1, playerId);
        if (score > bestScore) { bestScore = score; bestAction = action; }
    }
    return bestAction;
}

function minimax(env, depth, playerId) {
    if (env.done) {
        const w = env.info.winner;
        if (w === -1) return 0;
        return w === playerId ? 10 - depth : depth - 10;
    }
    if (env.currentPlayer === playerId) {
        let best = -Infinity;
        for (const a of env.getValidActions()) {
            const s = env.clone(); s.step(a);
            best = Math.max(best, minimax(s, depth + 1, playerId));
        }
        return best;
    } else {
        let best = Infinity;
        for (const a of env.getValidActions()) {
            const s = env.clone(); s.step(a);
            best = Math.min(best, minimax(s, depth + 1, playerId));
        }
        return best;
    }
}

// ============================================================
// Neural network helpers
// ============================================================

function processState(state, rows, cols) {
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

function softmax(arr) {
    let max = -Infinity;
    for (let i = 0; i < arr.length; i++) if (arr[i] > max) max = arr[i];
    const exp = new Float32Array(arr.length);
    let sum = 0;
    for (let i = 0; i < arr.length; i++) { exp[i] = Math.exp(arr[i] - max); sum += exp[i]; }
    for (let i = 0; i < arr.length; i++) exp[i] /= sum;
    return exp;
}

async function onnxPredict(session, state, rows, cols) {
    const input = processState(state, rows, cols);
    const tensor = new ort.Tensor("float32", input, [1, 2, rows, cols]);
    const results = await session.run({ input: tensor });
    const policy = softmax(results.policy.data);
    const value = results.value.data[0];
    return [policy, value];
}

// ============================================================
// Pure MCTS (UCT + rollouts) — port of giotto/agents/algorithms/mcts.py
// ============================================================

class MCTSNode {
    constructor(env, parent, parentAction, playerJustMoved) {
        this.env = env;
        this.parent = parent;
        this.parentAction = parentAction;
        this.playerJustMoved = playerJustMoved;
        this.children = new Map();
        this.untriedActions = env.getValidActions();
        this.totalVisits = 0;
        this.totalValue = 0.0;
    }

    get avgValue() {
        return this.totalVisits === 0 ? 0.0 : this.totalValue / this.totalVisits;
    }

    isFullyExpanded() {
        return this.untriedActions.length === 0;
    }

    bestChild(cpuct) {
        const playerToMove = 1 - this.playerJustMoved;
        let bestScore = -Infinity;
        let bestEntries = [];

        for (const [action, child] of this.children) {
            const exploitation = child.playerJustMoved === playerToMove
                ? child.avgValue
                : -child.avgValue;
            const exploration = cpuct * Math.sqrt(Math.log(this.totalVisits) / child.totalVisits);
            const ucb = exploitation + exploration;

            if (ucb > bestScore) {
                bestScore = ucb;
                bestEntries = [[action, child]];
            } else if (ucb === bestScore) {
                bestEntries.push([action, child]);
            }
        }
        return bestEntries[Math.floor(Math.random() * bestEntries.length)];
    }
}

function mctsRollout(env, rootPlayer) {
    while (!env.done) {
        const actions = env.getValidActions();
        env.step(actions[Math.floor(Math.random() * actions.length)]);
    }
    const winner = env.info.winner;
    if (winner === -1) return 0;
    return winner === rootPlayer ? 1 : -1;
}

function mctsBackpropagate(node, result, rootPlayer) {
    while (node !== null) {
        node.totalVisits++;
        if (node.playerJustMoved === rootPlayer) {
            node.totalValue += result;
        } else {
            node.totalValue -= result;
        }
        node = node.parent;
    }
}

function mctsAction(env, nSimulations, cpuct) {
    const rootPlayer = env.currentPlayer;
    const rootEnv = env.clone();
    const root = new MCTSNode(rootEnv, null, null, 1 - rootEnv.currentPlayer);

    for (let i = 0; i < nSimulations; i++) {
        let node = root;
        const simEnv = env.clone();

        // SELECTION
        while (!simEnv.done && node.isFullyExpanded() && node.children.size > 0) {
            const [action, child] = node.bestChild(cpuct);
            simEnv.step(action);
            node = child;
        }

        // EXPANSION
        if (!simEnv.done && node.untriedActions.length > 0) {
            const idx = Math.floor(Math.random() * node.untriedActions.length);
            const action = node.untriedActions[idx];
            node.untriedActions.splice(idx, 1);

            const playerJustMoved = simEnv.currentPlayer;
            simEnv.step(action);

            const child = new MCTSNode(simEnv.clone(), node, action, playerJustMoved);
            node.children.set(action, child);
            node = child;
        }

        // SIMULATION
        const result = mctsRollout(simEnv, rootPlayer);

        // BACKPROPAGATION
        mctsBackpropagate(node, result, rootPlayer);
    }

    // Most visited child
    let bestAction = null, bestVisits = -1;
    for (const [action, child] of root.children) {
        if (child.totalVisits > bestVisits) {
            bestVisits = child.totalVisits;
            bestAction = action;
        }
    }
    return bestAction;
}

// ============================================================
// Async MCTS
// ============================================================

class AZNode {
    constructor(env, parent, parentAction, sourceEnv, sourceAction) {
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

    expand() {
        if (this.children.size > 0 || this.env.done) return;
        for (const action of this.env.getValidActions()) {
            this.children.set(action, new AZNode(null, this, action, this._env, action));
        }
    }

    selectChild(cpuct) {
        const sqrtP = Math.sqrt(this.nVisits + 1);
        const cs = cpuct * sqrtP;
        let best = null, bestScore = -1e9;
        for (const child of this.children.values()) {
            const score = -child.avgValue + cs * child.prob / (child.nVisits + 1);
            if (score > bestScore) { bestScore = score; best = child; }
        }
        return best;
    }

    backpropagate(value) {
        this.totalScore += value;
        this.nVisits++;
        if (this.parent !== null) this.parent.backpropagate(-value);
    }

    isLeaf() { return this.children.size === 0; }
    isTerminal() { return this.env.done; }

    terminalEval() {
        const w = this.env.info.winner;
        if (w === -1) return 0.0;
        return w === this.toPlay ? 1.0 : -1.0;
    }
}

async function runMCTS(env, session, rows, cols, nSimulations, cpuct) {
    // Build root
    const rootEnv = env.clone();
    const root = new AZNode(rootEnv, null, null, null, null);
    const validActions = rootEnv.getValidActions();
    const [rootPolicy] = await onnxPredict(session, rootEnv.getState(), rows, cols);

    // Set root children with policy priors
    let probSum = 0;
    for (const action of validActions) {
        const child = new AZNode(null, root, action, rootEnv, action);
        child.prob = rootPolicy[action - 1];
        probSum += child.prob;
        root.children.set(action, child);
    }
    for (const child of root.children.values()) child.prob /= probSum;

    // Run simulations
    for (let sim = 0; sim < nSimulations; sim++) {
        let node = root;

        while (!node.isTerminal() && !node.isLeaf()) {
            node = node.selectChild(cpuct);
        }

        let value;
        if (!node.isTerminal()) {
            node.expand();
            const [policy, val] = await onnxPredict(session, node.env.getState(), rows, cols);
            const nodeActions = node.env.getValidActions();
            let sum = 0;
            for (const a of nodeActions) sum += policy[a - 1];
            for (const a of nodeActions) {
                node.children.get(a).prob = policy[a - 1] / sum;
            }
            value = val;
        } else {
            value = node.terminalEval();
        }

        node.backpropagate(value);
    }

    // Select best action (greedy — most visited)
    let bestAction = null, bestVisits = -1;
    for (const [action, child] of root.children) {
        if (child.nVisits > bestVisits) { bestVisits = child.nVisits; bestAction = action; }
    }
    return bestAction;
}

// ============================================================
// Worker message handler
// ============================================================

const ENV_MAP = { tris: TrisEnv, connect4: Connect4Env };
const DIMS = { tris: [3, 3], connect4: [6, 7] };

let currentSession = null;
let currentGame = null;

function reconstructEnv(game, board, currentPlayer, moves) {
    const env = new ENV_MAP[game]();
    env.board = Int8Array.from(board);
    env.currentPlayer = currentPlayer;
    env.turnCounter = moves.length;
    env.done = false;
    env.info = { moves: [...moves] };
    return env;
}

self.onmessage = async function (e) {
    const msg = e.data;

    if (msg.type === "init") {
        currentGame = msg.game;
        try {
            if (msg.modelUrl) {
                currentSession = await ort.InferenceSession.create(msg.modelUrl, {
                    executionProviders: ["wasm"],
                });
            }
            self.postMessage({ type: "ready" });
        } catch (err) {
            self.postMessage({ type: "error", message: err.message });
        }
    } else if (msg.type === "move") {
        const { board, currentPlayer, moves, agent, simulations } = msg;
        const env = reconstructEnv(currentGame, board, currentPlayer, moves);

        let action;
        if (agent === "random") {
            action = randomAction(env);
        } else if (agent === "minimax") {
            action = minimaxAction(env);
        } else if (agent === "mcts") {
            const cpuct = 1.4;
            const sims = simulations || (currentGame === "tris" ? 1000 : 500);
            action = mctsAction(env, sims, cpuct);
        } else if (agent === "alphazero") {
            const [rows, cols] = DIMS[currentGame];
            const cpuct = currentGame === "tris" ? 1.5 : 3.5;
            const sims = simulations || (currentGame === "tris" ? 100 : 800);
            action = await runMCTS(env, currentSession, rows, cols, sims, cpuct);
        }

        self.postMessage({ type: "action", action });
    }
};
