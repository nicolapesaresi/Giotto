/**
 * Main game controller — state machine: LAUNCHER -> MENU -> GAME -> GAMEOVER -> MENU
 */

import { TrisEnv } from "./engine/env-tris.js";
import { Connect4Env } from "./engine/env-connect4.js";
import { Launcher } from "./ui/launcher.js";
import { Menu } from "./ui/menu.js";
import { TrisBoard } from "./ui/board-tris.js";
import { Connect4Board } from "./ui/board-connect4.js";
import { GameOver } from "./ui/gameover.js";

const canvas = document.getElementById("game-canvas");
const loading = document.getElementById("loading");

// Resize canvas to fill viewport while maintaining 4:3 aspect ratio
function resizeCanvas() {
    const maxW = window.innerWidth;
    const maxH = window.innerHeight;
    const ratio = 4 / 3;

    let cssW, cssH;
    if (maxW / maxH > ratio) {
        cssH = maxH;
        cssW = cssH * ratio;
    } else {
        cssW = maxW;
        cssH = cssW / ratio;
    }

    canvas.style.width = `${cssW}px`;
    canvas.style.height = `${cssH}px`;

    // Scale canvas buffer to device pixel ratio for sharp rendering
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(cssW * dpr);
    canvas.height = Math.round(cssH * dpr);
}
window.addEventListener("resize", resizeCanvas);
resizeCanvas();

// ============================================================
// State
// ============================================================

let currentGame = null; // "tris" | "connect4"
let env = null;
let board = null;
let worker = null;
let xAgent = null;
let oAgent = null;
let workerReady = false;
let aiThinking = false;

function getAgents() {
    return {
        0: oAgent,
        1: xAgent,
    };
}

function needsWorker(agent) {
    return agent.id !== "human";
}

function needsModel(agent) {
    return agent.id === "alphazero";
}

// ============================================================
// Worker management
// ============================================================

function initWorker(game, modelUrl) {
    return new Promise((resolve, reject) => {
        if (worker) worker.terminate();
        worker = new Worker("js/worker/ai-worker.js");
        workerReady = false;

        worker.onmessage = (e) => {
            if (e.data.type === "ready") {
                workerReady = true;
                resolve();
            } else if (e.data.type === "error") {
                reject(new Error(e.data.message));
            } else if (e.data.type === "action") {
                handleAIMove(e.data.action);
            }
        };

        worker.postMessage({ type: "init", game, modelUrl });
    });
}

function requestAIMove(agent) {
    aiThinking = true;
    showThinking(true, agent.label);
    worker.postMessage({
        type: "move",
        board: Array.from(env.board),
        currentPlayer: env.currentPlayer,
        moves: env.info.moves,
        agent: agent.id,
    });
}

function handleAIMove(action) {
    aiThinking = false;
    showThinking(false);
    env.step(action);
    board.draw();

    if (env.done) {
        showGameOver();
    } else {
        nextTurn();
    }
}

// ============================================================
// Game flow
// ============================================================

function showThinking(show, agentLabel = null) {
    loading.textContent = agentLabel ? `${agentLabel} is thinking...` : "AI is thinking...";
    loading.style.display = show ? "block" : "none";
}

function showLauncher() {
    showThinking(false);
    const launcher = new Launcher(canvas, (game) => {
        currentGame = game;
        showMenu();
    });
    launcher.show();
}

function showMenu() {
    const menu = new Menu(canvas, currentGame, async (x, o) => {
        xAgent = x;
        oAgent = o;
        await startGame();
    }, () => {
        showLauncher();
    });
    menu.show();
}

async function startGame() {
    // Create environment
    env = currentGame === "tris" ? new TrisEnv() : new Connect4Env();
    env.reset(1); // X always starts

    // Init worker if any AI agent is selected
    if (needsWorker(xAgent) || needsWorker(oAgent)) {
        showThinking(true);
        const ctx = canvas.getContext("2d");
        ctx.fillStyle = "rgb(30,30,30)";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = "#aaa";
        ctx.font = `${Math.floor(canvas.height * 0.04)}px monospace`;
        ctx.textAlign = "center";
        const loadMsg = (needsModel(xAgent) || needsModel(oAgent))
            ? "Loading AI model..." : "Initializing...";
        ctx.fillText(loadMsg, canvas.width / 2, canvas.height / 2);

        try {
            const modelUrl = (needsModel(xAgent) || needsModel(oAgent))
                ? new URL(`models/${currentGame}.onnx`, window.location.href).href
                : null;
            await initWorker(currentGame, modelUrl);
        } catch (err) {
            console.error("Failed to load AI model:", err);
            ctx.fillStyle = "red";
            ctx.fillText("Failed to load model: " + err.message, canvas.width / 2, canvas.height * 0.6);
            return;
        }
        showThinking(false);
    }

    // Create board renderer
    const agents = getAgents();
    const BoardClass = currentGame === "tris" ? TrisBoard : Connect4Board;
    board = new BoardClass(canvas, env, agents, (action) => {
        // Human move callback
        if (aiThinking || env.done) return;
        env.step(action);
        board.draw();

        if (env.done) {
            board.disableInput();
            showGameOver();
        } else {
            nextTurn();
        }
    });

    board.draw();
    nextTurn();
}

function nextTurn() {
    const cp = env.currentPlayer;
    const agent = getAgents()[cp];

    if (agent.id === "human") {
        board.enableInput();
    } else {
        board.disableInput();
        // Small delay so the board renders before AI starts thinking
        setTimeout(() => requestAIMove(agent), 50);
    }
}

function showGameOver() {
    board.disableInput();
    const gameover = new GameOver(canvas, env, getAgents(), () => {
        showMenu();
    });
    gameover.show();
}

// ============================================================
// Start
// ============================================================

showLauncher();
