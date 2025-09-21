// sdk/agentClient.js
// ES module. Requires global io() from socket.io.min.js loaded by the page.

export class AgentClient {
	/**
	 * @param {{ url?: string, path?: string }} opts
	 */
	constructor(opts = {}) {
		this.url = opts.url ?? window.location.origin;
		this.path = opts.path ?? "/socket.io";
		this.socket = null;

		this._buffer = "";
		this._activeRunId = null;
		this._runResolve = null;
		this._runReject = null;

		this._connectedOnce = false;
		this._lastReconnectAttempt = 0;
		this._onReconnect = null;

		this._cb = {
			onStarted: null,
			onChunk: null,
			onText: null,
			onDone: null,
			onError: null
		};

		this._global = {
			onStarted: null,
			onChunk: null,
			onText: null,
			onDone: null,
			onError: null
		};
	}

	/**
	 * Connect to the server. Optionally pass { onReconnect } to be notified
	 * when a reconnection succeeds after a drop.
	 *
	 * @param {{ onReconnect?: (attempt:number)=>void }=} options
	 * @returns {Promise<void>}
	 */
	async connect(options = {}) {
		if (this.socket) return;
		this._onReconnect = typeof options.onReconnect === "function" ? options.onReconnect : null;

		this.socket = io(this.url, {
			path: this.path,
			transports: ["websocket"],
			reconnection: true,
			reconnectionAttempts: Infinity,
			reconnectionDelay: 500,
			reconnectionDelayMax: 5000,
			timeout: 20000
		});

		// --- lifecycle
		this.socket.on("connect", () => {
			if (this._connectedOnce && this._onReconnect) {
				try { this._onReconnect(this._lastReconnectAttempt); } catch {}
			}
		});
		this.socket.on("connect_error", (err) => {
			if (!this._connectedOnce) return;
			console.debug("[AgentClient] connect_error:", err?.message || err);
		});
		this.socket.on("reconnect", (attempt) => { this._lastReconnectAttempt = attempt; });
		this.socket.on("reconnect_attempt", (attempt) => { this._lastReconnectAttempt = attempt; });
		this.socket.on("reconnect_error", (err) => { console.debug("[AgentClient] reconnect_error:", err?.message || err); });
		this.socket.on("reconnect_failed", () => { console.warn("[AgentClient] reconnect_failed"); });

		// --- streaming
		this.socket.on("RunStarted", (payload) => {
			this._activeRunId = payload?.runId ?? null;
			const h = this._cb.onStarted || this._global.onStarted;
			if (typeof h === "function") { try { h(this._activeRunId); } catch {} }
		});

		this.socket.on("ChatChunk", (payload) => {
			const piece = typeof payload?.chunk === "string" ? payload.chunk : "";
			if (!piece) return;
			this._buffer += piece;

			const hChunk = this._cb.onChunk || this._global.onChunk;
			const hText  = this._cb.onText  || this._global.onText;

			if (typeof hChunk === "function") { try { hChunk(piece); } catch {} }
			if (typeof hText  === "function") { try { hText(this._buffer); } catch {} }
		});

		this.socket.on("ChatDone", () => {
			const h = this._cb.onDone || this._global.onDone;
			if (typeof h === "function") { try { h(); } catch {} }
			if (this._runResolve) this._runResolve({ runId: this._activeRunId, text: this._buffer });
			this._clearRunState();
		});

		this.socket.on("Interrupted", () => {
			const err = { code: "INTERRUPTED", message: "Run interrupted" };
			const h = this._cb.onError || this._global.onError;
			if (typeof h === "function") { try { h(err); } catch {} }
			if (this._runReject) this._runReject(Object.assign(new Error(err.message), { code: err.code }));
			this._clearRunState();
		});

		this.socket.on("Error", (payload) => {
			const err = {
				code: payload?.code || "ERROR",
				message: payload?.message || "Unknown error",
				runId: payload?.runId ?? null
			};
			const h = this._cb.onError || this._global.onError;
			if (typeof h === "function") { try { h(err); } catch {} }
			if (this._runReject) this._runReject(Object.assign(new Error(err.message), { code: err.code }));
			this._clearRunState();
		});

		// Wait for initial connect or fail
		await new Promise((resolve, reject) => {
			const ok = () => { this.socket.off("connect_error", ko); resolve(); };
			const ko = (e) => { this.socket.off("connect", ok); reject(e); };
			this.socket.once("connect", ok);
			this.socket.once("connect_error", ko);
		});
		this._connectedOnce = true;
	}

	disconnect() {
		if (!this.socket) return;
		try { this.socket.disconnect(); } catch {}
		this.socket = null;
		this._clearRunState();
	}

	/**
	 * Register global stream handlers that fire when there is no active per-run
	 * callback set by runText().
	 *
	 * @param {{ onStarted?:(runId:string)=>void, onChunk?:(piece:string)=>void, onText?:(full:string)=>void, onDone?:()=>void, onError?:(err:{code:string,message:string,runId?:string|null})=>void }} cbs
	 */
	onStream(cbs = {}) {
		this._global.onStarted = typeof cbs.onStarted === "function" ? cbs.onStarted : null;
		this._global.onChunk   = typeof cbs.onChunk   === "function" ? cbs.onChunk   : null;
		this._global.onText    = typeof cbs.onText    === "function" ? cbs.onText    : null;
		this._global.onDone    = typeof cbs.onDone    === "function" ? cbs.onDone    : null;
		this._global.onError   = typeof cbs.onError   === "function" ? cbs.onError   : null;
	}

	/**
	 * Start a chat run.
	 *
	 * @param {string} text
	 * @param {{ agent:string, threadId?:string }} options
	 * @param {{ onStarted?:(runId:string)=>void, onChunk?:(piece:string)=>void, onText?:(full:string)=>void, onDone?:()=>void, onError?:(err:{code:string,message:string,runId?:string|null})=>void }} cbs
	 * @returns {Promise<{ runId:string|null, text:string }>}
	 */
	runText(text, options, cbs = {}) {
		if (!this.socket || !this.socket.connected) {
			return Promise.reject(Object.assign(new Error("Not connected"), { code: "NOT_CONNECTED" }));
		}
		if (typeof text !== "string" || !text.length) {
			return Promise.reject(Object.assign(new Error("Text is required"), { code: "BAD_ARGS" }));
		}
		if (!options || typeof options.agent !== "string" || !options.agent.length) {
			return Promise.reject(Object.assign(new Error("Agent is required"), { code: "BAD_ARGS" }));
		}

		this._cb.onStarted = typeof cbs.onStarted === "function" ? cbs.onStarted : null;
		this._cb.onChunk   = typeof cbs.onChunk   === "function" ? cbs.onChunk   : null;
		this._cb.onText    = typeof cbs.onText    === "function" ? cbs.onText    : null;
		this._cb.onDone    = typeof cbs.onDone    === "function" ? cbs.onDone    : null;
		this._cb.onError   = typeof cbs.onError   === "function" ? cbs.onError   : null;

		this._buffer = "";
		this._activeRunId = null;

		const payload = {
			text,
			agent: options.agent,
			thread_id: options.threadId || null
		};

		return new Promise((resolve, reject) => {
			this._runResolve = resolve;
			this._runReject = reject;
			try {
				this.socket.emit("Chat", payload);
			} catch (e) {
				this._runReject = null;
				this._runResolve = null;
				reject(Object.assign(new Error("Emit failed"), { code: "EMIT_FAILED", cause: e }));
			}
		});
	}

	/**
	 * Interrupt the current run (if any).
	 */
	cancel() {
		if (!this.socket || !this.socket.connected) return;
		try { this.socket.emit("Interrupt", { runId: this._activeRunId ?? null }); } catch {}
	}

	/**
	 * Ask the agent server to subscribe to transcripts from the STT server
	 * on behalf of this client connection (server-side multiplex).
	 *
	 * @param {{ sttUrl:string, clientId:string, agent:string, threadId?:string }} args
	 * @returns {Promise<void>}
	 */
	sttSubscribe(args) {
		if (!this.socket || !this.socket.connected) {
			return Promise.reject(Object.assign(new Error("Not connected"), { code: "NOT_CONNECTED" }));
		}
		const { sttUrl, clientId, agent, threadId } = args || {};
		if (!sttUrl || !clientId || !agent) {
			return Promise.reject(Object.assign(new Error("sttUrl, clientId, and agent are required"), { code: "BAD_ARGS" }));
		}
		return new Promise((resolve, reject) => {
			try {
				this.socket.emit("JoinSTT", {
					sttUrl,
					clientId,
					agent,
					threadId: threadId || null
				}, (ack) => {
					if (ack && ack.error) return reject(Object.assign(new Error(ack.error), { code: "STT_SUBSCRIBE_ERROR" }));
					resolve();
				});
			} catch (e) {
				reject(Object.assign(new Error("Emit failed"), { code: "EMIT_FAILED", cause: e }));
			}
		});
	}

	/**
	 * Unsubscribe from server-side STT multiplex stream.
	 *
	 * @param {{ sttUrl:string, clientId:string }} args
	 * @returns {Promise<void>}
	 */
	sttUnsubscribe(args) {
		if (!this.socket || !this.socket.connected) {
			return Promise.reject(Object.assign(new Error("Not connected"), { code: "NOT_CONNECTED" }));
		}
		const { sttUrl, clientId } = args || {};
		if (!sttUrl || !clientId) {
			return Promise.reject(Object.assign(new Error("sttUrl and clientId are required"), { code: "BAD_ARGS" }));
		}
		return new Promise((resolve, reject) => {
			try {
				this.socket.emit("LeaveSTT", { sttUrl, clientId }, (ack) => {
					if (ack && ack.error) return reject(Object.assign(new Error(ack.error), { code: "STT_UNSUBSCRIBE_ERROR" }));
					resolve();
				});
			} catch (e) {
				reject(Object.assign(new Error("Emit failed"), { code: "EMIT_FAILED", cause: e }));
			}
		});
	}

	get activeRunId() {
		return this._activeRunId;
	}

	_clearRunState() {
		this._activeRunId = null;
		this._runResolve = null;
		this._runReject = null;
		this._buffer = "";
		this._cb = { onStarted: null, onChunk: null, onText: null, onDone: null, onError: null };
	}
}
