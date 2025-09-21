// audioResampler.js
// ES module; tabs for indentation.

export class AudioResampler {
	/**
	 * @param {number} inRate  - input sample rate (e.g., 48000)
	 * @param {number} outRate - output sample rate (e.g., 16000)
	 */
	constructor(inRate, outRate) {
		this._ratio = inRate / outRate;
		this._carry = new Float32Array(0);
	}

	/**
	 * Push a mono Float32Array chunk at inRate; returns Int16Array at outRate (or null if not enough data yet).
	 * @param {Float32Array} chunk
	 * @returns {Int16Array|null}
	 */
	pushFloat32(chunk) {
		// Concatenate the carry and the new chunk.
		const input = new Float32Array(this._carry.length + chunk.length);
		input.set(this._carry, 0);
		input.set(chunk, this._carry.length);

		// Number of output samples we can generate.
		const outLen = Math.floor(input.length / this._ratio);

		// Not enough data yet: save input as new carry and wait for more.
		if (outLen === 0) {
			this._carry = input;
			return null;
		}

		const out = new Int16Array(outLen);
		let pos = 0; // last input index used (retained from your code; not used elsewhere)

		// Linear interpolation resampling.
		for (let i = 0; i < outLen; i++) {
			const idx = i * this._ratio;
			const i0 = Math.floor(idx);
			const i1 = Math.min(i0 + 1, input.length - 1);
			const frac = idx - i0;

			// y = y0 * (1 - x) + y1 * x
			const sample = input[i0] * (1 - frac) + input[i1] * frac;

			// Clamp to [-1,1] and convert to 16-bit PCM
			const s = Math.max(-1, Math.min(1, sample));
			out[i] = (s < 0 ? s * 0x8000 : s * 0x7FFF) | 0;

			pos = i0;
		}

		// Save the remainder for the next call.
		const remainderStart = Math.floor(outLen * this._ratio);
		this._carry = input.subarray(remainderStart);

		return out;
	}

	/** Reset internal carry buffer. */
	reset() {
		this._carry = new Float32Array(0);
	}
}
