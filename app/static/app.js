/* Tabs = 4 spaces */

/* ---------- DOM helpers ---------- */
function $(id) { return document.getElementById(id); }
function scrollToBottom(el) { el.scrollTop = el.scrollHeight; }

/* ---------- Elements ---------- */
const messagesEl = $('messages');
const eventsEl = $('eventsLog');
const textEl = $('text');
const sendBtn = $('sendBtn');
const stopBtn = $('stopBtn');
const agentEl = $('agentSelect');
const memoryModeEl = $('memoryMode');
const newThreadBtn = $('newThreadBtn');

/* ---------- UUID helper ---------- */
function genId() {
	// Prefer native UUID, fallback to random string
	if (window.crypto && typeof window.crypto.randomUUID === 'function') {
		return window.crypto.randomUUID();
	}
	return 't_' + Math.random().toString(36).slice(2) + Date.now().toString(36);
}

/* ---------- Thread state (always present) ---------- */
let currentThreadId = genId();

/* ---------- Socket.IO ---------- */
const socket = io({ path: '/socket.io' });

/* ---------- Event logging (separate pane) ---------- */
function logEvent(line) {
	const ts = new Date().toISOString().split('T')[1].replace('Z', '');
	eventsEl.textContent += `[${ts}] ${line}\n`;
	scrollToBottom(eventsEl);
}

/* ---------- Message rendering ---------- */
let currentRunId = null;
let currentAssistantMsg = null;

function addUserMessage(text) {
	const d = document.createElement('div');
	d.className = 'msg user';
	d.textContent = text;
	messagesEl.appendChild(d);
	scrollToBottom(messagesEl);
}

function startAssistantMessage(runId) {
	currentRunId = runId;
	currentAssistantMsg = document.createElement('div');
	currentAssistantMsg.className = 'msg assistant';
	currentAssistantMsg.dataset.runId = runId;
	currentAssistantMsg.textContent = '';
	messagesEl.appendChild(currentAssistantMsg);
	scrollToBottom(messagesEl);
}

function appendAssistantChunk(runId, chunk) {
	if (!currentAssistantMsg || currentRunId !== runId) {
		startAssistantMessage(runId);
	}
	currentAssistantMsg.textContent += chunk;
	scrollToBottom(messagesEl);
}

function endAssistantMessage() {
	currentRunId = null;
	currentAssistantMsg = null;
}

/* ---------- Socket.IO handlers ---------- */
socket.on('connect', () => {
	logEvent(`[sio] connected: ${socket.id}`);
	logEvent(`[thread] initialized ${currentThreadId}`);
});

socket.on('disconnect', (reason) => {
	logEvent(`[sio] disconnected: ${reason}`);
});

socket.on('RunStarted', (payload) => {
	const { runId } = payload || {};
	logEvent(`Run started (${runId}) thread=${currentThreadId}`);
	startAssistantMessage(runId);
});

socket.on('ChatChunk', (payload) => {
	const { runId, chunk } = payload || {};
	if (typeof chunk === 'string' && runId) {
		appendAssistantChunk(runId, chunk);
	}
});

socket.on('ChatDone', (payload) => {
	const { runId } = payload || {};
	logEvent(`[done] (${runId})`);
	endAssistantMessage();
});

socket.on('Interrupted', (payload) => {
	const { runId } = payload || {};
	logEvent(`[interrupted] (${runId || '-'})`);
	endAssistantMessage();
});

socket.on('Error', (payload) => {
	const msg = (payload && payload.message) ? payload.message : 'Unknown error';
	logEvent(`[error] ${msg}`);
	endAssistantMessage();
});

/* ---------- UI actions ---------- */
sendBtn.addEventListener('click', () => {
	const text = (textEl.value || '').trim();
	if (!text) return;

	// immediately reflect user message and clear the input
	addUserMessage(text);
	textEl.value = '';

	// build payload (always include thread_id)
	const agent = agentEl.value || 'router';
	const memoryMode = memoryModeEl.value || 'none';

	const data = {
		agent,
		text,
		thread_id: currentThreadId,
		memory: memoryMode === 'none'
			? 'none'
			: { mode: 'thread_window' }, // add parameters later if needed
	};

	socket.emit('Chat', data);
});

stopBtn.addEventListener('click', () => {
	socket.emit('Interrupt', {});
});

newThreadBtn.addEventListener('click', () => {
	// Generate new id, clear UI, log
	currentThreadId = genId();
	messagesEl.innerHTML = '';
	endAssistantMessage();
	logEvent(`[thread] new ${currentThreadId}`);
});

/* Enter => send */
textEl.addEventListener('keydown', (e) => {
	if (e.key === 'Enter' && !e.shiftKey) {
		e.preventDefault();
		sendBtn.click();
	}
});
