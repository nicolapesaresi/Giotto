/**
 * Tic Tac Toe environment — port of giotto/envs/tris.py
 * Board: Int8Array(9), values: -1 (empty), 0 (O), 1 (X)
 * Actions: 1-9 (row-major: 1=top-left, 9=bottom-right)
 */

export class TrisEnv {
    constructor() {
        this.rows = 3;
        this.cols = 3;
        this.signs = ["o", "x", " "];
        this.board = new Int8Array(9);
        this.reset(1);
    }

    reset(startingPlayer = null) {
        if (startingPlayer !== 0 && startingPlayer !== 1) {
            startingPlayer = Math.random() < 0.5 ? 0 : 1;
        }
        this.currentPlayer = startingPlayer;
        this.board.fill(-1);
        this.turnCounter = 0;
        this.done = false;
        this.info = { moves: [] };
    }

    step(action) {
        this.info.moves.push(action);
        const [row, col] = this.decodeAction(action);
        this.board[row * 3 + col] = this.currentPlayer;
        this.turnCounter++;

        if (this.checkWin(this.currentPlayer)) {
            this.done = true;
            this.info.winner = this.currentPlayer;
        } else if (this.turnCounter === 9) {
            this.done = true;
            this.info.winner = -1;
        }
        this.currentPlayer = 1 - this.currentPlayer;
    }

    decodeAction(action) {
        const idx = action - 1;
        return [Math.floor(idx / 3), idx % 3];
    }

    getValidActions() {
        const actions = [];
        for (let i = 0; i < 9; i++) {
            if (this.board[i] === -1) actions.push(i + 1);
        }
        return actions;
    }

    getState() {
        return [Int8Array.from(this.board), this.currentPlayer];
    }

    checkWin(player) {
        const b = this.board;
        // rows
        for (let r = 0; r < 3; r++) {
            const i = r * 3;
            if (b[i] === player && b[i + 1] === player && b[i + 2] === player) return true;
        }
        // cols
        for (let c = 0; c < 3; c++) {
            if (b[c] === player && b[c + 3] === player && b[c + 6] === player) return true;
        }
        // diagonals
        if (b[0] === player && b[4] === player && b[8] === player) return true;
        if (b[2] === player && b[4] === player && b[6] === player) return true;
        return false;
    }

    getWinningCells(player) {
        const b = this.board;
        // rows
        for (let r = 0; r < 3; r++) {
            const i = r * 3;
            if (b[i] === player && b[i + 1] === player && b[i + 2] === player) {
                return [[r, 0], [r, 1], [r, 2]];
            }
        }
        // cols
        for (let c = 0; c < 3; c++) {
            if (b[c] === player && b[c + 3] === player && b[c + 6] === player) {
                return [[0, c], [1, c], [2, c]];
            }
        }
        // diagonals
        if (b[0] === player && b[4] === player && b[8] === player) {
            return [[0, 0], [1, 1], [2, 2]];
        }
        if (b[2] === player && b[4] === player && b[6] === player) {
            return [[0, 2], [1, 1], [2, 0]];
        }
        return null;
    }

    clone() {
        const env = Object.create(TrisEnv.prototype);
        env.rows = 3;
        env.cols = 3;
        env.signs = this.signs;
        env.board = Int8Array.from(this.board);
        env.currentPlayer = this.currentPlayer;
        env.turnCounter = this.turnCounter;
        env.done = this.done;
        env.info = { moves: [...this.info.moves] };
        if ("winner" in this.info) env.info.winner = this.info.winner;
        return env;
    }
}
