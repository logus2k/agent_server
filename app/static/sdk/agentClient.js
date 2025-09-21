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

		this._buffer = "";				// aggregated text for onText()
		this._activeRunId = null;		// last RunStarted id
		this._runResolve = null;		// pending run promise resolve
		this._runReject = null;			// pending run promise reject
		this._connectedOnce = false;

		// last callbacks passed to runText
		this._cb = {
			onStarted: null,
			onChunk: null,
			onText: null,
			onDone: null,
			onError: null
		};
	}

	/**
	 * Connect to the server. You can pass { onReconnect } to be notified when
	 * a reconnection succeeds after a drop.
	 *
	 * @param {{ onReconnect?: (attempt:number)=>void }=} options
	 * @returns {Promise<void>}
	 */
	async connect(options = {}) {
		if (this.socket && this.socket.connected) return;

		// Configure built-in exponential backoff reconnection.
		this.socket = io(this.url, {
			path: this.path,
			transports: ["websocket", "polling"],
			reconnection: true,
			reconnectionAttempts: Infinity,
			reconnectionDelay: 500,
			reconnectionDelayMax: 5000,
			randomizationFactor: 0.5,
			autoConnect: true
		});

		const { onReconnect } = options;

		this.socket.on("connect", () => {
			// First connection OR successful reconnect
			if (this._connectedOnce && typeof onReconnect === "function") {
				onReconnect(this._lastReconnectAttempt ?? 0);
			}
			this._connectedOnce = true;
		});

		this.socket.on("reconnect_attempt", (attempt) => {
			this._lastReconnectAttempt = attempt;
		});

		this.socket.on("reconnect_error", (err) => {
			// Surface as console noise only; SDK user can hook connect().catch if needed
			console.debug("[AgentClient] reconnect_error:", err?.message || err);
		});

		this.socket.on("reconnect_failed", () => {
			console.warn("[AgentClient] reconnect_failed");
		});

		// ---- Stream handlers ----
		this.socket.on("RunStarted", (payload) => {
			this._activeRunId = payload?.runId ?? null;
			if (typeof this._cb.onStarted === "function") {
				try { this._cb.onStarted(this._activeRunId); } catch {}
			}
		});

		this.socket.on("ChatChunk", (payload) => {
			const piece = (payload && typeof payload.chunk === "string") ? payload.chunk : "";
			if (!piece) return;
			this._buffer += piece;

			if (typeof this._cb.onChunk === "function") {
				try { this._cb.onChunk(piece); } catch {}
			}
			if (typeof this._cb.onText === "function") {
				try { this._cb.onText(this._buffer); } catch {}
			}
		});

		this.socket.on("ChatDone", (_payload) => {
			try {
				if (typeof this._cb.onDone === "function") this._cb.onDone();
				if (this._runResolve) this._runResolve({ runId: this._activeRunId, text: this._buffer });
			} finally {
				this._clearRunState();
			}
		});

		this.socket.on("Interrupted", (_payload) => {
			try {
				if (typeof this._cb.onError === "function") this._cb.onError({ code: "INTERRUPTED", message: "Run interrupted" });
				if (this._runReject) this._runReject(Object.assign(new Error("Interrupted"), { code: "INTERRUPTED" }));
			} finally {
				this._clearRunState();
			}
		});

		this.socket.on("Error", (payload) => {
			const err = {
				code: payload?.code || "ERROR",
				message: payload?.message || "Unknown error",
				runId: payload?.runId ?? null
			};
			try {
				if (typeof this._cb.onError === "function") this._cb.onError(err);
				if (this._runReject) this._runReject(Object.assign(new Error(err.message), { code: err.code }));
			} finally {
				this._clearRunState();
			}
		});

		// Wait until first 'connect'
		await new Promise((resolve, reject) => {
			const ok = () => {
				this.socket.off("connect_error", ko);
				resolve();
			};
			const ko = (e) => {
				this.socket.off("connect", ok);
				reject(e);
			};
			this.socket.once("connect", ok);
			this.socket.once("connect_error", ko);
		});
	}

	/**
	 * Run a text prompt with an agent.
	 * @param {string} text
	 * @param {{ agent: string, threadId?: string }} options
	 * @param {{
	 * 	onStarted?: (runId:string)=>void,
	 * 	onChunk?: (delta:string)=>void,
	 * 	onText?: (full:string)=>void,
	 * 	onDone?: ()=>void,
	 * 	onError?: (err:{code:string, message:string, runId?:string|null})=>void
	 * }} cbs
	 * @returns {Promise<{ runId: string|null, text: string }>}
	 */
	async runText(text, options, cbs = {}) {
		if (!this.socket || !this.socket.connected) {
			throw new Error("Socket is not connected");
		}
		if (this._activeRunId) {
			throw new Error("A run is already in progress for this client");
		}

		this._buffer = "";
		this._cb = {
			onStarted: cbs.onStarted || null,
			onChunk: cbs.onChunk || null,
			onText: cbs.onText || null,
			onDone: cbs.onDone || null,
			onError: cbs.onError || null
		};

		// Fire the request
		this.socket.emit("Chat", {
			agent: options?.agent,
			text: text,
			// Server controls memory; threadId is still OK if server expects it
			thread_id: options?.threadId
		});

		return new Promise((resolve, reject) => {
			this._runResolve = resolve;
			this._runReject = reject;
			// Optional: timeout safety could be added here if desired
		});
	}

	/**
	 * Cancel the in-flight run. If runId is provided, it is checked
	 * against the active one; if omitted, cancels whatever is active.
	 * @param {string=} runId
	 * @returns {boolean} true if an Interrupt was sent
	 */
	cancel(runId) {
		if (!this.socket) return false;
		if (!this._activeRunId) return false;
		if (runId && runId !== this._activeRunId) return false;

		this.socket.emit("Interrupt");
		return true;
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
