/**
 * Player selection menu — choose agents for X and O before starting a game.
 * Layout matches Pygame main_menu.py positioning.
 */

const AGENT_TYPES = {
    tris: [
        { id: "human", label: "Human" },
        { id: "random", label: "Random" },
        { id: "minimax", label: "Minimax" },
        { id: "mcts", label: "MCTS" },
        { id: "alphazero", label: "Giottino" },
    ],
    connect4: [
        { id: "human", label: "Human" },
        { id: "random", label: "Random" },
        { id: "mcts", label: "MCTS" },
        { id: "alphazero", label: "Giotto" },
    ],
};

export class Menu {
    constructor(canvas, game, onStart, onBack) {
        this.canvas = canvas;
        this.ctx = canvas.getContext("2d");
        this.game = game;
        this.onStart = onStart;
        this.onBack = onBack;

        const types = AGENT_TYPES[game];
        this.agents = types;
        this.xIndex = 0;  // Human
        this.oIndex = types.length - 1;  // AlphaZero
        this._buttons = [];
        this._bound = this._onClick.bind(this);
    }

    show() {
        this.canvas.addEventListener("click", this._bound);
        this.canvas.addEventListener("touchstart", this._bound, { passive: false });
        this.draw();
    }

    hide() {
        this.canvas.removeEventListener("click", this._bound);
        this.canvas.removeEventListener("touchstart", this._bound);
    }

    draw() {
        const ctx = this.ctx;
        const w = this.canvas.width;
        const h = this.canvas.height;

        ctx.fillStyle = "rgb(30,30,30)";
        ctx.fillRect(0, 0, w, h);

        // Title — matches Pygame TitleText
        const title = this.game === "tris" ? "Tic-Tac-Toe" : "Connect 4";
        ctx.fillStyle = "#ffff00";
        ctx.font = `bold ${Math.floor(h * 0.08)}px monospace`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(title, w / 2, h * 0.12);

        this._buttons = [];

        const btnW = w * 0.375;
        const btnH = h * 0.1;
        const btnX = (w - btnW) / 2;

        // START button at 30%
        const startW = w * 0.2;
        const startH = h * 0.1;
        this._addButton((w - startW) / 2, h * 0.28, startW, startH,
            "START", "start", "#ffff00", "#000");

        // X player selector at 45%
        const xLabel = `x player: ${this.agents[this.xIndex].label}`;
        this._addButton(btnX, h * 0.43, btnW, btnH,
            xLabel, "x_select", "rgb(60,30,30)");

        // O player selector at 58%
        const oLabel = `o player: ${this.agents[this.oIndex].label}`;
        this._addButton(btnX, h * 0.57, btnW, btnH,
            oLabel, "o_select", "rgb(25,55,25)");

        // Swap button — to the right, between the two player rows
        const swapW = w * 0.1;
        const swapH = h * 0.055;
        const swapX = w * 0.75 - swapW / 2;
        const swapY = h * 0.55 - swapH / 2;
        ctx.fillStyle = "rgb(60,60,100)";
        ctx.fillRect(swapX, swapY, swapW, swapH);
        ctx.fillStyle = "rgb(200,200,255)";
        ctx.font = `${Math.floor(swapH * 0.55)}px monospace`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText("swap", swapX + swapW / 2, swapY + swapH / 2);
        this._buttons.push({ x: swapX, y: swapY, w: swapW, h: swapH, action: "swap" });

        // Back button
        const backW = w * 0.2;
        const backH = h * 0.05;
        this._addButton((w - backW) / 2, h * 0.72, backW, backH,
            "Back", "back", "rgb(40,40,40)");
    }

    _addButton(x, y, w, h, label, action, bg, fg = "#fff") {
        const ctx = this.ctx;
        ctx.fillStyle = bg;
        ctx.fillRect(x, y, w, h);
        ctx.fillStyle = fg;
        ctx.font = `${Math.floor(h * 0.42)}px monospace`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(label, x + w / 2, y + h / 2);
        this._buttons.push({ x, y, w, h, action });
    }

    _onClick(e) {
        e.preventDefault();
        const pos = this._getPos(e);
        for (const btn of this._buttons) {
            if (pos.x >= btn.x && pos.x <= btn.x + btn.w &&
                pos.y >= btn.y && pos.y <= btn.y + btn.h) {
                this._handleAction(btn.action);
                return;
            }
        }
    }

    _handleAction(action) {
        if (action === "x_select") {
            this.xIndex = (this.xIndex + 1) % this.agents.length;
            this.draw();
        } else if (action === "o_select") {
            this.oIndex = (this.oIndex + 1) % this.agents.length;
            this.draw();
        } else if (action === "swap") {
            [this.xIndex, this.oIndex] = [this.oIndex, this.xIndex];
            this.draw();
        } else if (action === "start") {
            this.hide();
            this.onStart(this.agents[this.xIndex], this.agents[this.oIndex]);
        } else if (action === "back") {
            this.hide();
            this.onBack();
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
