/**
 * Canvas renderer for 3x3 Tic-Tac-Toe board.
 * Layout matches Pygame: grid from 20%-80% of canvas.
 */

export class TrisBoard {
    constructor(canvas, env, agents, onMove, onMenu) {
        this.canvas = canvas;
        this.ctx = canvas.getContext("2d");
        this.env = env;
        this.agents = agents; // {0: agentObj, 1: agentObj}
        this.onMove = onMove; // callback when human clicks a valid cell
        this.onMenu = onMenu; // callback when menu button is clicked
        this._bound = this._onClick.bind(this);
        this._boundMenu = this._onMenuClick.bind(this);
        this._menuBtn = null;
        this._inputEnabled = false;
        // Menu button listener is always active
        this.canvas.addEventListener("click", this._boundMenu);
        this.canvas.addEventListener("touchstart", this._boundMenu, { passive: false });
    }

    enableInput() {
        this._inputEnabled = true;
        this.canvas.addEventListener("click", this._bound);
        this.canvas.addEventListener("touchstart", this._bound, { passive: false });
    }

    disableInput() {
        this._inputEnabled = false;
        this.canvas.removeEventListener("click", this._bound);
        this.canvas.removeEventListener("touchstart", this._bound);
    }

    destroy() {
        this.disableInput();
        this.canvas.removeEventListener("click", this._boundMenu);
        this.canvas.removeEventListener("touchstart", this._boundMenu);
    }

    draw() {
        const ctx = this.ctx;
        const w = this.canvas.width;
        const h = this.canvas.height;

        // Background
        ctx.fillStyle = "rgb(30,30,30)";
        ctx.fillRect(0, 0, w, h);

        // Grid coordinates (matching Pygame settings)
        const x0 = w * 0.2, x1 = w * 0.4, x2 = w * 0.6, x3 = w * 0.8;
        const y0 = h * 0.2, y1 = h * 0.4, y2 = h * 0.6, y3 = h * 0.8;
        const lineW = Math.max(2, w * 0.01);

        // Grid lines
        ctx.strokeStyle = "white";
        ctx.lineWidth = lineW;

        ctx.beginPath();
        ctx.moveTo(x1, y0); ctx.lineTo(x1, y3);
        ctx.moveTo(x2, y0); ctx.lineTo(x2, y3);
        ctx.moveTo(x0, y1); ctx.lineTo(x3, y1);
        ctx.moveTo(x0, y2); ctx.lineTo(x3, y2);
        ctx.stroke();

        // Cell dimensions
        const cellW = x1 - x0;
        const cellH = y1 - y0;

        // Draw pieces
        for (let cell = 1; cell <= 9; cell++) {
            const idx = cell - 1;
            const player = this.env.board[idx];
            if (player === -1) continue;

            const row = Math.floor(idx / 3);
            const col = idx % 3;
            const cx = x0 + col * cellW + cellW / 2;
            const cy = y0 + row * cellH + cellH / 2;
            const sign = this.env.signs[player];

            ctx.font = `bold ${Math.floor(cellH * 0.6)}px monospace`;
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillStyle = sign === "x" ? "red" : "#00ff00";
            ctx.fillText(sign, cx, cy);
        }

        // Winning line
        const winner = this.env.info.winner;
        if (this.env.done && winner !== undefined && winner !== -1) {
            const cells = this.env.getWinningCells(winner);
            if (cells) {
                const start = this._cellCenter(cells[0], x0, y0, cellW, cellH);
                const end = this._cellCenter(cells[cells.length - 1], x0, y0, cellW, cellH);
                ctx.strokeStyle = "#ffff00";
                ctx.lineWidth = 5;
                ctx.beginPath();
                ctx.moveTo(start.x, start.y);
                ctx.lineTo(end.x, end.y);
                ctx.stroke();
            }
        }

        // Turn text
        this._drawTurnText();

        // In-game menu button (top-left)
        this._drawMenuButton();
    }

    _cellCenter([row, col], x0, y0, cellW, cellH) {
        return {
            x: x0 + col * cellW + cellW / 2,
            y: y0 + row * cellH + cellH / 2,
        };
    }

    _drawTurnText() {
        const ctx = this.ctx;
        const w = this.canvas.width;
        const h = this.canvas.height;
        const cp = this.env.currentPlayer;
        const sign = this.env.signs[cp];
        const name = this.agents[cp].label;

        ctx.font = `${Math.floor(h * 0.035)}px monospace`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";

        const text = `Turn ${this.env.turnCounter + 1} - `;
        const signText = sign;
        const nameText = ` (${name}) to move`;

        // Measure parts
        const textW = ctx.measureText(text).width;
        const signW = ctx.measureText(signText).width;
        const nameW = ctx.measureText(nameText).width;
        const totalW = textW + signW + nameW;
        const startX = (w - totalW) / 2;
        const y = h * 0.1;

        ctx.textAlign = "left";
        ctx.fillStyle = "white";
        ctx.fillText(text, startX, y);
        ctx.fillStyle = sign === "x" ? "red" : "#00ff00";
        ctx.fillText(signText, startX + textW, y);
        ctx.fillStyle = "white";
        ctx.fillText(nameText, startX + textW + signW, y);
    }

    _drawMenuButton() {
        const ctx = this.ctx;
        const w = this.canvas.width;
        const h = this.canvas.height;

        const btnW = w * 0.1;
        const btnH = h * 0.055;
        const btnX = w * 0.08 - btnW / 2;
        const btnY = h * 0.05 - btnH / 2;

        ctx.fillStyle = "#ffff00";
        ctx.fillRect(btnX, btnY, btnW, btnH);
        ctx.fillStyle = "#000";
        ctx.font = `bold ${Math.floor(btnH * 0.5)}px monospace`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText("MENU", btnX + btnW / 2, btnY + btnH / 2);

        this._menuBtn = { x: btnX, y: btnY, w: btnW, h: btnH };
    }

    _onMenuClick(e) {
        const pos = this._getPos(e);
        if (this._menuBtn) {
            const b = this._menuBtn;
            if (pos.x >= b.x && pos.x <= b.x + b.w && pos.y >= b.y && pos.y <= b.y + b.h) {
                e.preventDefault();
                this.destroy();
                this.onMenu();
            }
        }
    }

    _onClick(e) {
        e.preventDefault();
        const pos = this._getPos(e);

        const w = this.canvas.width;
        const h = this.canvas.height;
        const x0 = w * 0.2, cellW = w * 0.2;
        const y0 = h * 0.2, cellH = h * 0.2;

        const col = Math.floor((pos.x - x0) / cellW);
        const row = Math.floor((pos.y - y0) / cellH);

        if (row < 0 || row > 2 || col < 0 || col > 2) return;

        const cell = row * 3 + col + 1;
        if (this.env.getValidActions().includes(cell)) {
            this.onMove(cell);
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
