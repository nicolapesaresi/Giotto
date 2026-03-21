/**
 * Connect 4 environment — port of giotto/envs/connect4.py
 * Board: Int8Array(42), 6 rows x 7 cols, row 0 = bottom
 * Values: -1 (empty), 0 (O), 1 (X)
 * Actions: 1-7 (column number)
 */

export class Connect4Env {
    constructor() {
        this.rows = 6;
        this.cols = 7;
        this.signs = ["o", "x", " "];
        this.board = new Int8Array(42);
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
        this.board[row * 7 + col] = this.currentPlayer;
        this.turnCounter++;

        if (this.checkWin(this.currentPlayer)) {
            this.done = true;
            this.info.winner = this.currentPlayer;
        } else if (this.turnCounter === 42) {
            this.done = true;
            this.info.winner = -1;
        }
        this.currentPlayer = 1 - this.currentPlayer;
    }

    decodeAction(action) {
        const col = action - 1;
        // find lowest empty row in this column (gravity)
        for (let row = 0; row < 6; row++) {
            if (this.board[row * 7 + col] === -1) {
                return [row, col];
            }
        }
        throw new Error(`Column ${action} is full`);
    }

    getValidActions() {
        const actions = [];
        for (let c = 0; c < 7; c++) {
            // column playable if any cell is empty
            for (let r = 0; r < 6; r++) {
                if (this.board[r * 7 + c] === -1) {
                    actions.push(c + 1);
                    break;
                }
            }
        }
        return actions;
    }

    getState() {
        return [Int8Array.from(this.board), this.currentPlayer];
    }

    checkWin(player) {
        const b = this.board;
        // horizontal
        for (let r = 0; r < 6; r++) {
            for (let c = 0; c <= 3; c++) {
                const i = r * 7 + c;
                if (b[i] === player && b[i + 1] === player && b[i + 2] === player && b[i + 3] === player) return true;
            }
        }
        // vertical
        for (let r = 0; r <= 2; r++) {
            for (let c = 0; c < 7; c++) {
                const i = r * 7 + c;
                if (b[i] === player && b[i + 7] === player && b[i + 14] === player && b[i + 21] === player) return true;
            }
        }
        // diagonal down-right (\) — but since row 0=bottom, this is visually /
        for (let r = 0; r <= 2; r++) {
            for (let c = 0; c <= 3; c++) {
                const i = r * 7 + c;
                if (b[i] === player && b[i + 8] === player && b[i + 16] === player && b[i + 24] === player) return true;
            }
        }
        // diagonal down-left (/) — visually \
        for (let r = 0; r <= 2; r++) {
            for (let c = 3; c < 7; c++) {
                const i = r * 7 + c;
                if (b[i] === player && b[i + 6] === player && b[i + 12] === player && b[i + 18] === player) return true;
            }
        }
        return false;
    }

    getWinningCells(player) {
        const b = this.board;
        // horizontal
        for (let r = 0; r < 6; r++) {
            for (let c = 0; c <= 3; c++) {
                const i = r * 7 + c;
                if (b[i] === player && b[i + 1] === player && b[i + 2] === player && b[i + 3] === player) {
                    return [[r, c], [r, c + 1], [r, c + 2], [r, c + 3]];
                }
            }
        }
        // vertical
        for (let r = 0; r <= 2; r++) {
            for (let c = 0; c < 7; c++) {
                const i = r * 7 + c;
                if (b[i] === player && b[i + 7] === player && b[i + 14] === player && b[i + 21] === player) {
                    return [[r, c], [r + 1, c], [r + 2, c], [r + 3, c]];
                }
            }
        }
        // diagonal up-right
        for (let r = 0; r <= 2; r++) {
            for (let c = 0; c <= 3; c++) {
                const i = r * 7 + c;
                if (b[i] === player && b[i + 8] === player && b[i + 16] === player && b[i + 24] === player) {
                    return [[r, c], [r + 1, c + 1], [r + 2, c + 2], [r + 3, c + 3]];
                }
            }
        }
        // diagonal up-left
        for (let r = 0; r <= 2; r++) {
            for (let c = 3; c < 7; c++) {
                const i = r * 7 + c;
                if (b[i] === player && b[i + 6] === player && b[i + 12] === player && b[i + 18] === player) {
                    return [[r, c], [r + 1, c - 1], [r + 2, c - 2], [r + 3, c - 3]];
                }
            }
        }
        return null;
    }

    clone() {
        const env = Object.create(Connect4Env.prototype);
        env.rows = 6;
        env.cols = 7;
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
