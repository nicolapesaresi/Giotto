/**
 * Launcher screen — game selection (Tic-Tac-Toe or Connect 4).
 */

export class Launcher {
    constructor(canvas, onSelect) {
        this.canvas = canvas;
        this.ctx = canvas.getContext("2d");
        this.onSelect = onSelect;
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

        // Title
        ctx.fillStyle = "#ffff00";
        ctx.font = `bold ${Math.floor(h * 0.12)}px monospace`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText("Giotto AI", w / 2, h * 0.22);

        ctx.fillStyle = "#aaa";
        ctx.font = `${Math.floor(h * 0.035)}px monospace`;
        ctx.fillText("Select a game", w / 2, h * 0.34);

        const btnW = w * 0.45;
        const btnH = h * 0.09;
        const btnX = (w - btnW) / 2;

        this._buttons = [
            { x: btnX, y: h * 0.44, w: btnW, h: btnH, game: "tris", label: "Tic-Tac-Toe" },
            { x: btnX, y: h * 0.57, w: btnW, h: btnH, game: "connect4", label: "Connect 4" },
        ];

        for (const btn of this._buttons) {
            ctx.fillStyle = "#ffff00";
            ctx.fillRect(btn.x, btn.y, btn.w, btn.h);
            ctx.fillStyle = "#000";
            ctx.font = `bold ${Math.floor(btnH * 0.45)}px monospace`;
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText(btn.label, btn.x + btn.w / 2, btn.y + btn.h / 2);
        }
    }

    _onClick(e) {
        e.preventDefault();
        const pos = this._getPos(e);
        for (const btn of this._buttons) {
            if (pos.x >= btn.x && pos.x <= btn.x + btn.w &&
                pos.y >= btn.y && pos.y <= btn.y + btn.h) {
                this.hide();
                this.onSelect(btn.game);
                return;
            }
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
