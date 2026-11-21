/**
 * Wire codecs for Amazon Quick (SNCS) WebSocket transports:
 * - AWS EventStream binary framing + CRC32
 * - Minimal CBOR encoder/decoder for Gateway frames
 * - AIDP JSON/base64 frame codec & Gateway CBOR frame codec
 */

const enc = new TextEncoder()
const dec = new TextDecoder()

const crc32Table = (() => {
  const table = new Uint32Array(256)
  for (let i = 0; i < 256; i++) {
    let c = i
    for (let j = 0; j < 8; j++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1
    table[i] = c >>> 0
  }
  return table
})()

/** Compute standard ISO 3309 CRC-32 checksum over bytes. */
export function crc32(data) {
  let crc = 0xffffffff
  for (let i = 0; i < data.length; i++) {
    crc = crc32Table[(crc ^ data[i]) & 0xff] ^ (crc >>> 8)
  }
  return (crc ^ 0xffffffff) >>> 0
}

const HV_BOOL_TRUE = 0
const HV_BOOL_FALSE = 1
const HV_BYTE = 2
const HV_SHORT = 3
const HV_INT = 4
const HV_LONG = 5
const HV_BYTES = 6
const HV_STRING = 7
const HV_TIMESTAMP = 8
const HV_UUID = 9

/**
 * Encode string headers and binary payload into an AWS EventStream binary frame.
 * @param {Record<string, string>} headers
 * @param {Uint8Array} payload
 * @returns {Uint8Array}
 */
export function esEncode(headers, payload) {
  const hb = []
  for (const [name, value] of Object.entries(headers)) {
    const nb = enc.encode(name)
    hb.push(nb.length)
    for (let i = 0; i < nb.length; i++) hb.push(nb[i])
    hb.push(HV_STRING)
    const vb = enc.encode(String(value))
    hb.push((vb.length >> 8) & 0xff, vb.length & 0xff)
    for (let i = 0; i < vb.length; i++) hb.push(vb[i])
  }
  const headersLen = hb.length
  const totalLen = 12 + headersLen + payload.length + 4
  const msg = new Uint8Array(totalLen)
  const view = new DataView(msg.buffer)
  view.setUint32(0, totalLen, false)
  view.setUint32(4, headersLen, false)
  view.setUint32(8, crc32(msg.subarray(0, 8)), false)
  msg.set(hb, 12)
  msg.set(payload, 12 + headersLen)
  view.setUint32(totalLen - 4, crc32(msg.subarray(0, totalLen - 4)), false)
  return msg
}

/**
 * Drain zero or more complete AWS EventStream messages from `buffer` starting at `offset`.
 * @param {Uint8Array} buffer
 * @param {number} [offset=0]
 * @returns {{ messages: Array<{ headers: Record<string, unknown>, payload: Uint8Array }>, consumed: number }}
 */
export function esDrain(buffer, offset = 0) {
  const messages = []
  const n = buffer.length
  let pos = offset
  while (pos + 16 <= n) {
    const view = new DataView(buffer.buffer, buffer.byteOffset + pos, n - pos)
    const totalLen = view.getUint32(0, false)
    if (totalLen < 16 || totalLen > 16 * 1024 * 1024) {
      pos += 1
      continue
    }
    if (pos + totalLen > n) break
    const headersLen = view.getUint32(4, false)
    const hstart = pos + 12
    const hend = hstart + headersLen
    if (hend > pos + totalLen - 4) {
      pos += 1
      continue
    }
    const headers = {}
    let j = hstart
    let ok = true
    while (j < hend) {
      const nl = buffer[j++]
      if (j + nl + 1 > hend) { ok = false; break }
      const name = dec.decode(buffer.subarray(j, j + nl))
      j += nl
      const vt = buffer[j++]
      if (vt === HV_BOOL_TRUE) {
        headers[name] = true
      } else if (vt === HV_BOOL_FALSE) {
        headers[name] = false
      } else if (vt === HV_BYTE) {
        headers[name] = buffer[j]
        j += 1
      } else if (vt === HV_SHORT) {
        headers[name] = (buffer[j] << 8) | buffer[j + 1]
        j += 2
      } else if (vt === HV_INT) {
        headers[name] = new DataView(buffer.buffer, buffer.byteOffset + j, 4).getUint32(0, false)
        j += 4
      } else if (vt === HV_LONG || vt === HV_TIMESTAMP) {
        let v = 0
        for (let k = 0; k < 8; k++) v = v * 256 + buffer[j + k]
        headers[name] = v
        j += 8
      } else if (vt === HV_BYTES) {
        const bl = (buffer[j] << 8) | buffer[j + 1]
        j += 2
        headers[name] = buffer.slice(j, j + bl)
        j += bl
      } else if (vt === HV_STRING) {
        const vl = (buffer[j] << 8) | buffer[j + 1]
        j += 2
        headers[name] = dec.decode(buffer.subarray(j, j + vl))
        j += vl
      } else if (vt === HV_UUID) {
        headers[name] = buffer.slice(j, j + 16)
        j += 16
      } else {
        ok = false
        break
      }
    }
    if (ok) {
      const payload = buffer.slice(hend, pos + totalLen - 4)
      messages.push({ headers, payload })
    }
    pos += totalLen
  }
  return { messages, consumed: pos - offset }
}

function pushCborHead(out, major, len) {
  const m = major << 5
  if (len < 24) out.push(m | len)
  else if (len < 0x100) out.push(m | 24, len)
  else if (len < 0x10000) out.push(m | 25, (len >> 8) & 0xff, len & 0xff)
  else {
    out.push(m | 26, (len >>> 24) & 0xff, (len >>> 16) & 0xff, (len >>> 8) & 0xff, len & 0xff)
  }
}

/**
 * Encode a JS value (objects, arrays, strings, Uint8Array, integers, booleans, null) to CBOR.
 * @param {unknown} value
 * @returns {Uint8Array}
 */
export function cborEncode(value) {
  const out = []
  const write = (v) => {
    if (v === null || v === undefined) {
      out.push(v === null ? 0xf6 : 0xf7)
      return
    }
    if (typeof v === "boolean") {
      out.push(v ? 0xf5 : 0xf4)
      return
    }
    if (typeof v === "number" && Number.isInteger(v)) {
      if (v >= 0) pushCborHead(out, 0, v)
      else pushCborHead(out, 1, -1 - v)
      return
    }
    if (v instanceof Uint8Array) {
      pushCborHead(out, 2, v.length)
      for (let i = 0; i < v.length; i++) out.push(v[i])
      return
    }
    if (typeof v === "string") {
      const b = enc.encode(v)
      pushCborHead(out, 3, b.length)
      for (let i = 0; i < b.length; i++) out.push(b[i])
      return
    }
    if (Array.isArray(v)) {
      pushCborHead(out, 4, v.length)
      for (const item of v) write(item)
      return
    }
    if (typeof v === "object") {
      const entries = Object.entries(v).filter(([, val]) => val !== undefined)
      pushCborHead(out, 5, entries.length)
      for (const [k, val] of entries) {
        write(k)
        write(val)
      }
      return
    }
    throw new TypeError(`Unsupported CBOR type: ${typeof v}`)
  }
  write(value)
  return new Uint8Array(out)
}

/**
 * Decode a CBOR-encoded byte array into JS values.
 * @param {Uint8Array} bytes
 * @returns {unknown}
 */
export function cborDecode(bytes) {
  let pos = 0
  const readUint = (ai) => {
    if (ai < 24) return ai
    if (ai === 24) return bytes[pos++]
    if (ai === 25) {
      const v = (bytes[pos] << 8) | bytes[pos + 1]
      pos += 2
      return v
    }
    if (ai === 26) {
      const v =
        ((bytes[pos] << 24) >>> 0) +
        (bytes[pos + 1] << 16) +
        (bytes[pos + 2] << 8) +
        bytes[pos + 3]
      pos += 4
      return v
    }
    if (ai === 27) {
      let v = 0
      for (let i = 0; i < 8; i++) v = v * 256 + bytes[pos + i]
      pos += 8
      return v
    }
    if (ai === 31) return -1
    throw new Error(`Unsupported CBOR additional info: ${ai}`)
  }

  const read = () => {
    if (pos >= bytes.length) throw new Error("Unexpected end of CBOR input")
    const head = bytes[pos++]
    const major = head >> 5
    const ai = head & 0x1f
    if (major === 0) return readUint(ai)
    if (major === 1) return -1 - readUint(ai)
    if (major === 2) {
      if (ai === 31) {
        const chunks = []
        let total = 0
        while (bytes[pos] !== 0xff) {
          const c = read()
          chunks.push(c)
          total += c.length
        }
        pos += 1
        const merged = new Uint8Array(total)
        let off = 0
        for (const c of chunks) {
          merged.set(c, off)
          off += c.length
        }
        return merged
      }
      const len = readUint(ai)
      const slice = bytes.slice(pos, pos + len)
      pos += len
      return slice
    }
    if (major === 3) {
      if (ai === 31) {
        let s = ""
        while (bytes[pos] !== 0xff) s += read()
        pos += 1
        return s
      }
      const len = readUint(ai)
      const str = dec.decode(bytes.subarray(pos, pos + len))
      pos += len
      return str
    }
    if (major === 4) {
      if (ai === 31) {
        const arr = []
        while (bytes[pos] !== 0xff) arr.push(read())
        pos += 1
        return arr
      }
      const len = readUint(ai)
      const arr = new Array(len)
      for (let i = 0; i < len; i++) arr[i] = read()
      return arr
    }
    if (major === 5) {
      const obj = {}
      if (ai === 31) {
        while (bytes[pos] !== 0xff) {
          const k = String(read())
          obj[k] = read()
        }
        pos += 1
        return obj
      }
      const len = readUint(ai)
      for (let i = 0; i < len; i++) {
        const k = String(read())
        obj[k] = read()
      }
      return obj
    }
    if (major === 6) {
      readUint(ai)
      return read()
    }
    if (major === 7) {
      if (ai === 20) return false
      if (ai === 21) return true
      if (ai === 22) return null
      if (ai === 23) return undefined
      if (ai === 27) {
        const view = new DataView(bytes.buffer, bytes.byteOffset + pos, 8)
        pos += 8
        return view.getFloat64(0, false)
      }
      throw new Error(`Unsupported CBOR simple/float ai: ${ai}`)
    }
    throw new Error(`Unsupported CBOR major type: ${major}`)
  }

  return read()
}

/**
 * Encode an outbound AIDP WebSocket JSON payload (rawInputEvent wrapping base64 JSON).
 * @param {object} obj
 * @returns {Uint8Array}
 */
export function buildAidpInputFrame(obj) {
  const b64 = Buffer.from(JSON.stringify(obj), "utf8").toString("base64")
  return Buffer.from(JSON.stringify({ eventType: "rawInputEvent", payload: { payload: b64 } }), "utf8")
}

/**
 * Decode an inbound AIDP AWS EventStream frame into `{ type: "event", event }`, `{ type: "error", code, message }`, or `null`.
 * @param {Record<string, unknown>} headers
 * @param {Uint8Array} payload
 */
export function decodeAidpFrame(headers, payload) {
  const mt = headers[":message-type"]
  if (mt === "error" || mt === "exception") {
    const code = String(headers[":error-code"] || headers[":exception-type"] || "unknown")
    let message = String(headers[":error-message"] || "")
    if (!message && payload.length) {
      try {
        const parsed = JSON.parse(dec.decode(payload))
        message = parsed?.message || parsed?.Message || dec.decode(payload)
      } catch {
        message = dec.decode(payload)
      }
    }
    return { type: "error", code, message: message || code }
  }
  if (headers[":event-type"] !== "rawOutputEvent" || payload.length === 0) return null
  let outer
  try {
    outer = JSON.parse(dec.decode(payload))
  } catch {
    return null
  }
  const b64 = outer?.payload
  if (typeof b64 !== "string" || !b64) return null
  try {
    const event = JSON.parse(Buffer.from(b64, "base64").toString("utf8"))
    return { type: "event", event }
  } catch {
    return null
  }
}

/**
 * Encode an outbound Gateway WebSocket frame (AWS EventStream wrapping CBOR({ payload: utf8_json })).
 * @param {object} obj
 * @returns {Uint8Array}
 */
export function buildGatewayInputFrame(obj) {
  const innerCbor = cborEncode({ payload: enc.encode(JSON.stringify(obj)) })
  return esEncode(
    {
      ":event-type": "rawInputEvent",
      ":content-type": "application/cbor",
      ":message-type": "event",
    },
    innerCbor,
  )
}

/**
 * Decode an inbound Gateway AWS EventStream + CBOR frame.
 * @param {Record<string, unknown>} headers
 * @param {Uint8Array} payload
 */
export function decodeGatewayFrame(headers, payload) {
  const mt = headers[":message-type"]
  if (mt === "error" || mt === "exception") {
    const code = String(headers[":error-code"] || headers[":exception-type"] || "unknown")
    const message = String(headers[":error-message"] || "")
    return { type: "error", code, message: message ? `[${code}] ${message}` : code }
  }
  if (payload.length === 0) return null
  let decoded
  try {
    decoded = cborDecode(payload)
  } catch {
    return null
  }
  const inner = decoded?.payload
  if (!inner) return null
  const jsonStr = inner instanceof Uint8Array ? dec.decode(inner) : typeof inner === "string" ? inner : ""
  if (!jsonStr) return null
  try {
    return { type: "event", event: JSON.parse(jsonStr) }
  } catch {
    return null
  }
}
