/**
 * Game over — draws result text and "MENU" button directly on the game screen
 * without graying out the board. The board (with winning line) stays fully visible.
 */

export class GameOver {
    constructor(canvas, env, agents, onMenu) {
        this.canvas = canvas;
        this.ctx = canvas.getContext("2d");
        this.env = env;
        this.agents = agents;
        this.onMenu = onMenu;
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

        // Result text at the top (replaces turn text area)
        const winner = this.env.info.winner;
        let resultText, resultColor;

        if (winner === -1) {
            resultText = "Game over - Draw";
            resultColor = "white";
        } else {
            const sign = this.env.signs[winner];
            const name = this.agents[winner].label;
            resultText = `Game over - ${sign} (${name}) wins!`;
            resultColor = winner === 1 ? "red" : "#00ff00";
        }

        // Clear top area for result text
        ctx.fillStyle = "rgb(30,30,30)";
        ctx.fillRect(0, 0, w, h * 0.16);

        ctx.fillStyle = resultColor;
        ctx.font = `bold ${Math.floor(h * 0.045)}px monospace`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(resultText, w / 2, h * 0.08);

        // MENU button at the bottom
        const btnW = w * 0.2;
        const btnH = h * 0.06;
        const btnX = (w - btnW) / 2;
        const btnY = h * 0.91;

        // Clear button area
        ctx.fillStyle = "rgb(30,30,30)";
        ctx.fillRect(btnX - 2, btnY - 2, btnW + 4, btnH + 4);

        ctx.fillStyle = "#ffff00";
        ctx.fillRect(btnX, btnY, btnW, btnH);
        ctx.fillStyle = "#000";
        ctx.font = `bold ${Math.floor(btnH * 0.5)}px monospace`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText("MENU", btnX + btnW / 2, btnY + btnH / 2);

        this._buttons = [{ x: btnX, y: btnY, w: btnW, h: btnH }];
    }

    _onClick(e) {
        e.preventDefault();
        const pos = this._getPos(e);
        for (const btn of this._buttons) {
            if (pos.x >= btn.x && pos.x <= btn.x + btn.w &&
                pos.y >= btn.y && pos.y <= btn.y + btn.h) {
                this.hide();
                this.onMenu();
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
