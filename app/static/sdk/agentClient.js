// sdk/agentClient.js
export class AgentClient {
	constructor({ url, path = "/socket.io" } = {}) {
		this.url = url;
		this.path = path;
		this.socket = null;
		this.hello = null;
	}

	async connect() {
		if (this.socket) return this.hello;
		// io() is provided by socket.io.min.js
		this.socket = io(this.url, { path: this.path });
		await new Promise((resolve, reject) => {
			this.socket.on("connect", resolve);
			this.socket.on("connect_error", reject);
		});

		// Optional hello handshake (if your server emits it)
		this.hello = {
			server_version: "unknown",
			protocol_version: "1.0.0",
			agents: []
		};
		return this.hello;
	}

	/**
	 * Stream a text prompt to an agent.
	 * @param {string} text
	 * @param {{agent:string, threadId?:string}} meta
	 * @param {{
	 *   onStarted?:(runId:string)=>void,
	 *   onChunk?:(delta:string)=>void,
	 *   onText?:(fullText:string)=>void,
	 *   onDone?:()=>void,
	 *   onError?:(err:any)=>void,
	 *   signal?:AbortSignal
	 * }} callbacks
	 */
	async runText(text, meta, callbacks = {}) {
		await this.connect();
		const {
			onStarted, onChunk, onText, onDone, onError, signal
		} = callbacks;

		let runId = null;
		let buffer = "";

		const handleStarted = (payload) => {
			// server payload: { runId }
			runId = payload?.runId || null;
			onStarted?.(runId);
		};

		const handleChunk = (payload) => {
			// server payload: { runId, chunk: "<delta>" }
			const delta = typeof payload?.chunk === "string" ? payload.chunk : "";
			if (!delta) return;
			// Append to running buffer and notify
			buffer += delta;
			onChunk?.(delta);
			onText?.(buffer);
		};

		const handleDone = () => {
			cleanup();
			onDone?.();
			resolveRun?.();
		};

		const handleError = (payload) => {
			cleanup();
			const err = payload?.message || payload || "Unknown error";
			onError?.(err);
			rejectRun?.(new Error(String(err)));
		};

		const cleanup = () => {
			this.socket.off("RunStarted", handleStarted);
			this.socket.off("ChatChunk", handleChunk);
			this.socket.off("ChatDone", handleDone);
			this.socket.off("Error", handleError);
			if (signal) signal.removeEventListener("abort", onAbort);
		};

		const onAbort = () => {
			try { this.socket.emit("Interrupt"); } catch {}
		};

		if (signal) {
			if (signal.aborted) onAbort();
			else signal.addEventListener("abort", onAbort, { once: true });
		}

		let resolveRun, rejectRun;
		const donePromise = new Promise((res, rej) => {
			resolveRun = res;
			rejectRun = rej;
		});

		// Wire events
		this.socket.on("RunStarted", handleStarted);
		this.socket.on("ChatChunk", handleChunk);
		this.socket.on("ChatDone", handleDone);
		this.socket.on("Error", handleError);

		// Fire request
		this.socket.emit("Chat", {
			agent: meta?.agent,
			text,
			// server now controls memory policy; threadId is still useful for server-side memory
			thread_id: meta?.threadId || null
		});

		return donePromise;
	}
}
