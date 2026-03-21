/**
 * Canvas renderer for 6x7 Connect 4 board.
 * Layout matches Pygame: grid from 15%-85% horizontal, 25%-85% vertical.
 * Row 0 in env = bottom of visual board.
 */

export class Connect4Board {
    constructor(canvas, env, agents, onMove) {
        this.canvas = canvas;
        this.ctx = canvas.getContext("2d");
        this.env = env;
        this.agents = agents;
        this.onMove = onMove;
        this._bound = this._onClick.bind(this);
    }

    enableInput() {
        this.canvas.addEventListener("click", this._bound);
        this.canvas.addEventListener("touchstart", this._bound, { passive: false });
    }

    disableInput() {
        this.canvas.removeEventListener("click", this._bound);
        this.canvas.removeEventListener("touchstart", this._bound);
    }

    draw() {
        const ctx = this.ctx;
        const w = this.canvas.width;
        const h = this.canvas.height;

        ctx.fillStyle = "rgb(30,30,30)";
        ctx.fillRect(0, 0, w, h);

        // Grid bounds (matching Pygame settings_connect4.py proportions)
        const x0 = w * 0.15;
        const x7 = w * 0.85;
        const y0 = h * 0.25;
        const y6 = h * 0.85;
        const cellW = (x7 - x0) / 7;
        const cellH = (y6 - y0) / 6;
        const lineW = Math.max(2, w * 0.01);

        // Grid lines
        ctx.strokeStyle = "white";
        ctx.lineWidth = lineW;
        ctx.beginPath();

        // Vertical lines (left border, internal dividers, right border)
        for (let i = 0; i <= 7; i++) {
            const x = x0 + i * cellW;
            ctx.moveTo(x, y0);
            ctx.lineTo(x, y6);
        }
        // Horizontal lines (no top border, internal dividers + bottom border)
        for (let i = 1; i <= 6; i++) {
            const y = y0 + i * cellH;
            ctx.moveTo(x0, y);
            ctx.lineTo(x7, y);
        }
        ctx.stroke();

        // Draw pieces
        const radius = Math.min(cellW, cellH) * 0.35;
        for (let row = 0; row < 6; row++) {
            for (let col = 0; col < 7; col++) {
                const player = this.env.board[row * 7 + col];
                if (player === -1) continue;

                // Flip row for display (row 0 = bottom)
                const drawRow = 5 - row;
                const cx = x0 + col * cellW + cellW / 2;
                const cy = y0 + drawRow * cellH + cellH / 2;

                ctx.fillStyle = player === 1 ? "red" : "#00ff00";
                ctx.beginPath();
                ctx.arc(cx, cy, radius, 0, Math.PI * 2);
                ctx.fill();
            }
        }

        // Last move indicator
        if (this.env.info.moves.length > 0) {
            const lastCol = this.env.info.moves[this.env.info.moves.length - 1] - 1;
            let lastRow = -1;
            for (let r = 5; r >= 0; r--) {
                if (this.env.board[r * 7 + lastCol] !== -1) { lastRow = r; break; }
            }
            if (lastRow >= 0) {
                const drawRow = 5 - lastRow;
                ctx.strokeStyle = "rgb(160,160,160)";
                ctx.lineWidth = 2;
                ctx.strokeRect(
                    x0 + lastCol * cellW, y0 + drawRow * cellH,
                    cellW, cellH
                );
            }
        }

        // Winning line
        const winner = this.env.info.winner;
        if (this.env.done && winner !== undefined && winner !== -1) {
            const cells = this.env.getWinningCells(winner);
            if (cells) {
                const [r0, c0] = cells[0];
                const [r1, c1] = cells[cells.length - 1];
                const sx = x0 + c0 * cellW + cellW / 2;
                const sy = y0 + (5 - r0) * cellH + cellH / 2;
                const ex = x0 + c1 * cellW + cellW / 2;
                const ey = y0 + (5 - r1) * cellH + cellH / 2;
                ctx.strokeStyle = "#ffff00";
                ctx.lineWidth = 5;
                ctx.beginPath();
                ctx.moveTo(sx, sy);
                ctx.lineTo(ex, ey);
                ctx.stroke();
            }
        }

        // Turn text
        this._drawTurnText();
    }

    _drawTurnText() {
        const ctx = this.ctx;
        const w = this.canvas.width;
        const h = this.canvas.height;
        const cp = this.env.currentPlayer;
        const sign = this.env.signs[cp];
        const name = this.agents[cp].label;

        ctx.font = `${Math.floor(h * 0.035)}px monospace`;
        ctx.textBaseline = "middle";

        const text = `Turn ${this.env.turnCounter + 1} - `;
        const signText = sign;
        const nameText = ` (${name}) to move`;

        const textW = ctx.measureText(text).width;
        const signW = ctx.measureText(signText).width;
        const nameW = ctx.measureText(nameText).width;
        const totalW = textW + signW + nameW;
        const startX = (w - totalW) / 2;
        const y = h * 0.12;

        ctx.textAlign = "left";
        ctx.fillStyle = "white";
        ctx.fillText(text, startX, y);
        ctx.fillStyle = sign === "x" ? "red" : "#00ff00";
        ctx.fillText(signText, startX + textW, y);
        ctx.fillStyle = "white";
        ctx.fillText(nameText, startX + textW + signW, y);
    }

    _onClick(e) {
        e.preventDefault();
        const pos = this._getPos(e);
        const w = this.canvas.width;
        const x0 = w * 0.15;
        const cellW = (w * 0.85 - x0) / 7;

        const col = Math.floor((pos.x - x0) / cellW);
        if (col < 0 || col > 6) return;

        const action = col + 1;
        if (this.env.getValidActions().includes(action)) {
            this.onMove(action);
        }
    }

    _getPos(e) {
        const rect = this.canvas.getBoundingClientRect();
        const scaleX = this.canvas.width / rect.width;
        const scaleY = this.canvas.height / rect.height;
        if (e.touches) {
            return {
                x: (e.touches[0].clientX - rect.left) * scaleX,
                y: (e.touches[0].clientY - rect.top) * scaleY,
            };
        }
        return {
            x: (e.clientX - rect.left) * scaleX,
            y: (e.clientY - rect.top) * scaleY,
        };
    }
}
