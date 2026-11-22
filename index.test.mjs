import assert from "node:assert/strict"
import { createHash } from "node:crypto"
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
import test from "node:test"

import {
  buildAidpInputFrame,
  buildGatewayInputFrame,
  cborDecode,
  cborEncode,
  decodeAidpFrame,
  decodeGatewayFrame,
  esDrain,
  esEncode,
} from "./codec.mjs"
import {
  browserSignIn,
  desktopSignIn,
  importSignIns,
  invokeQbsApi,
  readChunkedCredential,
  readDesktop,
  renewOidc,
  signInsIn,
  usageFromProfile,
} from "./credentials.mjs"
import {
  configModels,
  createToolCallSplitter,
  executeSncsChat,
  modelsFromSettings,
  openaiToSncsRequest,
  resolveModelAndEffort,
} from "./bridge.mjs"
import { createAmazonQuickPlugin } from "./index.mjs"
import * as opencode1 from "./opencode.mjs"
import opencode2 from "./opencode2.mjs"

test("EventStream + CBOR + AIDP/Gateway codecs roundtrip cleanly", () => {
  const payload = new TextEncoder().encode('{"ok":true}')
  const frame = esEncode({ ":event-type": "rawOutputEvent", ":message-type": "event" }, payload)

  // Split across boundary to verify incremental drain
  const part1 = frame.subarray(0, 14)
  const part2 = frame.subarray(14)
  const d1 = esDrain(part1, 0)
  assert.equal(d1.messages.length, 0)
  assert.equal(d1.consumed, 0)

  const combined = new Uint8Array(part1.length + part2.length + frame.length)
  combined.set(part1, 0)
  combined.set(part2, part1.length)
  combined.set(frame, frame.length)
  const d2 = esDrain(combined, 0)
  assert.equal(d2.messages.length, 2)
  assert.equal(d2.consumed, combined.length)
  assert.equal(d2.messages[0].headers[":event-type"], "rawOutputEvent")

  // CBOR roundtrip
  const sample = { a: 1, b: -42, c: true, d: null, e: [1, "hi"], bytes: new Uint8Array([1, 2, 3]) }
  const decodedCbor = cborDecode(cborEncode(sample))
  assert.equal(decodedCbor.a, 1)
  assert.equal(decodedCbor.b, -42)
  assert.equal(decodedCbor.c, true)
  assert.equal(decodedCbor.d, null)
  assert.deepEqual(decodedCbor.e, [1, "hi"])
  assert.deepEqual(Array.from(decodedCbor.bytes), [1, 2, 3])

  // AIDP outbound & inbound decode
  const outAidp = JSON.parse(Buffer.from(buildAidpInputFrame({ type: "ping", payload: { n: 7 } })).toString("utf8"))
  assert.equal(outAidp.eventType, "rawInputEvent")
  assert.deepEqual(JSON.parse(Buffer.from(outAidp.payload.payload, "base64").toString("utf8")), { type: "ping", payload: { n: 7 } })

  const aidpErr = decodeAidpFrame(
    { ":message-type": "exception", ":exception-type": "AccessDeniedException" },
    new TextEncoder().encode(JSON.stringify({ message: "Unauthorized token" })),
  )
  assert.deepEqual(aidpErr, { type: "error", code: "AccessDeniedException", message: "Unauthorized token" })

  // Gateway outbound -> EventStream -> Gateway inbound roundtrip
  const gwFrame = buildGatewayInputFrame({ type: "text_delta", payload: { delta: "hello" } })
  const gwDrained = esDrain(gwFrame, 0)
  assert.equal(gwDrained.messages.length, 1)
  const gwDecoded = decodeGatewayFrame(gwDrained.messages[0].headers, gwDrained.messages[0].payload)
  assert.deepEqual(gwDecoded, { type: "event", event: { type: "text_delta", payload: { delta: "hello" } } })
})

test("readChunkedCredential & readDesktop decode UTF-16-LE chunked secrets read-only", () => {
  const fullJson = JSON.stringify({
    access_token: "acc-123",
    refresh_token: "ref-123",
    access_token_expiry: 1800000000.5,
    client_id: "cid-1",
    issuer_url: "https://identitycenter.amazonaws.com/ssoins-abc",
    tenant_url: "https://qbs-test.dp.appintegrations.us-east-1.prod.plato.ai.aws.dev",
    email: "user@example.com",
  })
  const half = Math.ceil(fullJson.length / 2)
  const c0 = fullJson.slice(0, half)
  const c1 = fullJson.slice(half)
  const sha = createHash("sha256").update(Buffer.from(fullJson, "utf16le")).digest("hex")
  const manifest = JSON.stringify({ __chunks__: 2, __sha__: sha })

  const store = new Map([
    ["quickwork-social-social-test", manifest],
    ["quickwork-social-social-test_chunk_0", c0],
    ["quickwork-social-social-test_chunk_1", c1],
  ])
  assert.equal(readChunkedCredential((k) => store.get(k) ?? null, "quickwork-social-social-test"), fullJson)

  // Corrupted hash returns null
  store.set("quickwork-social-social-test", JSON.stringify({ __chunks__: 2, __sha__: "0000" }))
  assert.equal(readChunkedCredential((k) => store.get(k) ?? null, "quickwork-social-social-test"), null)
  store.set("quickwork-social-social-test", manifest)

  // Temp profiles dir test
  const dir = mkdtempSync(join(tmpdir(), "aq-test-"))
  try {
    writeFileSync(
      join(dir, "profiles.json"),
      JSON.stringify({
        version: 1,
        last_active: "social-test",
        entries: [
          {
            id: "social-test",
            provider: "social",
            display_name: "Demo User",
            login_metadata: {
              issuer_url: "https://identitycenter.amazonaws.com/ssoins-abc",
              tenant_url: "https://qbs-test.dp.appintegrations.us-east-1.prod.plato.ai.aws.dev",
              client_id: "cid-1",
            },
          },
        ],
      }),
    )

    const readKeyring = (name) => readChunkedCredential((k) => store.get(k) ?? null, name)
    const desktop = readDesktop({ dir, readKeyring, noCache: true })
    assert.equal(desktop.access, "acc-123")
    assert.equal(desktop.expires, 1800000000500)
    assert.equal(desktop.region, "us-east-1")
    assert.equal(desktop.accountId, "user@example.com")
    assert.equal(desktop.source, "desktop")

    const authMethod = desktopSignIn({ dir, readKeyring })
    return authMethod.callback().then((res) => {
      assert.equal(res.type, "success")
      assert.equal(res.source, "desktop")
      assert.equal(res.access, "")
      assert.equal(res.refresh, "")
      assert.equal(res.accountId, "user@example.com")
    })
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

test("renewOidc deduplicates concurrent refreshes and handles importSignIns", async () => {
  let calls = 0
  const mockFetch = async (url, init) => {
    calls++
    assert.equal(url, "https://oidc.us-west-2.amazonaws.com/token")
    const body = JSON.parse(init.body)
    assert.equal(body.grantType, "refresh_token")
    assert.equal(body.refreshToken, "ref-old-1")
    return new Response(JSON.stringify({ accessToken: "acc-new", refreshToken: "ref-new", expiresIn: 3600 }), { status: 200 })
  }

  const baseAuth = {
    access: "acc-old",
    refresh: "ref-old-1",
    clientId: "cid-west",
    region: "us-west-2",
    tenantUrl: "https://qbs.dp.appintegrations.us-west-2.prod.plato.ai.aws.dev",
  }
  const [r1, r2] = await Promise.all([
    renewOidc(baseAuth, null, { fetch: mockFetch, now: 1_000_000 }),
    renewOidc(baseAuth, null, { fetch: mockFetch, now: 1_000_000 }),
  ])
  assert.equal(calls, 1)
  assert.equal(r1.access, "acc-new")
  assert.equal(r2.refresh, "ref-new")
  assert.equal(r1.expires, 1_000_000 + 3_600_000)

  // Import parser + renewal
  const parsed = signInsIn(
    JSON.stringify({
      access_token: "acc-exp",
      refresh_token: "ref-imp-2",
      client_id: "cid-west",
      tenant_url: "https://qbs.dp.appintegrations.us-west-2.prod.plato.ai.aws.dev",
    }),
  )
  assert.equal(parsed.length, 1)
  assert.equal(parsed[0].region, "us-west-2")

  const imported = await importSignIns(
    [
      JSON.stringify({
        access_token: "acc-exp",
        refresh_token: "ref-imp-2",
        client_id: "cid-west",
        tenant_url: "https://qbs.dp.appintegrations.us-west-2.prod.plato.ai.aws.dev",
      }),
    ],
    {
      fetch: async () => new Response(JSON.stringify({ accessToken: "acc-imp-fresh", refreshToken: "ref-imp-3", expiresIn: 1800 }), { status: 200 }),
    },
  )
  assert.equal(imported.length, 1)
  assert.equal(imported[0].type, "success")
  assert.equal(imported[0].access, "acc-imp-fresh")
})

test("invokeQbsApi and usageFromProfile support AIDP & Gateway transports", async () => {
  const aidpCalls = []
  const usageData = {
    monthly: { used: 25, limit: 100, label: "25 / 100 credits", percent: 25, resets_at: "2026-12-01T00:00:00Z" },
    session: { used: 2, limit: 20, label: "2 / 20 session", percent: 10, resume_in_minutes: 30 },
    overage_monthly: { used: 0, limit: 0, percent: 0 },
    subscription: { subscription_type: "FREE_TRIAL" },
  }
  const aidpOut = await invokeQbsApi(
    {
      providerType: "social",
      access: "tok-1",
      tenantUrl: "https://qbs-test.dp.appintegrations.us-east-1.prod.plato.ai.aws.dev",
    },
    "profile/usage",
    "GET",
    {},
    {
      fetch: async (url, init) => {
        aidpCalls.push({ url, body: JSON.parse(init.body) })
        return new Response(
          JSON.stringify({
            eventResponse: { invokeAgentResponse: JSON.stringify({ statusCode: 200, body: JSON.stringify(usageData) }) },
          }),
          { status: 200 },
        )
      },
    },
  )
  assert.equal(aidpCalls[0].url, "https://qbs-test.dp.appintegrations.us-east-1.prod.plato.ai.aws.dev/integration/quick-work/qbs-event")
  assert.equal(aidpCalls[0].body.eventBody.invokeAgent.path, "profile/usage")

  const magpieUsage = usageFromProfile(aidpOut)
  assert.equal(magpieUsage.plan, "Free Trial")
  assert.equal(magpieUsage.windows.length, 2)
  assert.equal(magpieUsage.windows[0].name, "Monthly usage")
  assert.equal(magpieUsage.windows[0].used, 25)
  assert.equal(magpieUsage.windows[0].resetsAt, "2026-12-01T00:00:00.000Z")
  assert.equal(magpieUsage.windows[1].name, "Session usage")

  // Gateway transport
  const gwUrls = []
  await invokeQbsApi(
    {
      providerType: "gateway",
      awsAccountId: "123456789012",
      access: "tok-gw",
      tenantUrl: "https://tenant.desktop.enterprise.quick.aws.com",
    },
    "api/ws-rpc",
    "POST",
    { type: "get_settings", payload: {} },
    {
      fetch: async (url) => {
        gwUrls.push(url)
        if (url.endsWith("/v1/discovery")) {
          return new Response(JSON.stringify({ cellHost: "cell1.desktop.enterprise.quick.aws.com" }), { status: 200 })
        }
        return new Response(JSON.stringify({ frames: [{ type: "settings", payload: { advanced_mode_enabled: true } }] }), { status: 200 })
      },
    },
  )
  assert.ok(gwUrls.some((u) => u.includes("cell1.desktop.enterprise.quick.aws.com/accounts/123456789012/invocations/rpc")))
})

test("model mapping, tool-call splitter, and openaiToSncsRequest handle variants and chunk boundaries", () => {
  const dynModels = modelsFromSettings("amazon-quick", {
    advanced_mode_enabled: true,
    model_slots: [{ id: "balanced", display_name: "Sonnet Custom", thinking: { mode: "optional", efforts: ["off", "high"], default_effort: "off" } }],
  })
  assert.equal(dynModels.balanced.name, "Sonnet Custom")
  assert.deepEqual(Object.keys(dynModels.balanced.variants), ["off", "high"])
  assert.ok(dynModels.advanced)

  assert.deepEqual(resolveModelAndEffort("amazon-quick/balanced@high"), {
    modelId: "balanced",
    agentMode: "balanced",
    autoRouting: false,
    thinkingEffort: "high",
  })
  assert.deepEqual(resolveModelAndEffort("fast", { reasoning_effort: "high" }), {
    modelId: "fast",
    agentMode: "fast",
    autoRouting: false,
    thinkingEffort: "off",
  })

  // OpenAI -> SNCS multi-turn + tools
  const sncs = openaiToSncsRequest({
    model: "smart",
    reasoning_effort: "medium",
    stream: true,
    tools: [{ type: "function", function: { name: "read_file", description: "Read", parameters: { type: "object" } } }],
    messages: [
      { role: "system", content: "Be concise." },
      { role: "user", content: "Check foo.txt" },
      { role: "assistant", content: null, tool_calls: [{ id: "c1", type: "function", function: { name: "read_file", arguments: '{"path":"foo.txt"}' } }] },
      { role: "tool", tool_call_id: "c1", content: "hello" },
    ],
  })
  assert.equal(sncs.userMessagePayload.agent_mode, "smart")
  assert.equal(sncs.userMessagePayload.thinking_effort, "medium")
  assert.equal(sncs.userMessagePayload.incognito, true)
  assert.equal(sncs.userMessagePayload.web_search_enabled, true)
  assert.equal(sncs.sessionConnectPayload.outpostId, undefined)
  assert.equal(sncs.userMessagePayload.outpostId, undefined)
  assert.equal(sncs.userMessagePayload.custom_prompt, null)
  assert.ok(sncs.userMessagePayload.text.includes("<instructions>"))
  assert.ok(sncs.userMessagePayload.text.includes("Be concise."))
  assert.ok(sncs.userMessagePayload.text.includes("CLIENT TOOL CALLING PROTOCOL"))
  assert.ok(sncs.userMessagePayload.text.includes('<tool_result id="c1">\nhello\n</tool_result>'))

  const sncsNoWeb = openaiToSncsRequest(
    { model: "fast", web_search_enabled: false, messages: [{ role: "user", content: "Ping" }] },
    { outpostId: "outpost-123" },
  )
  assert.equal(sncsNoWeb.userMessagePayload.web_search_enabled, false)
  assert.equal(sncsNoWeb.sessionConnectPayload.outpostId, "outpost-123")
  assert.equal(sncsNoWeb.userMessagePayload.outpostId, "outpost-123")

  // Stream splitter across partial tag prefixes
  const splitter = createToolCallSplitter(true)
  const chunks = ["Thinking... <", "t", "emp>not a tag</temp> <too", "l_call>{\"name\":\"read_file\",", "\"arguments\":{\"p\":1}}</tool", "_call> done."]
  let textOut = ""
  const calls = []
  for (const ch of chunks) {
    const res = splitter.push(ch)
    textOut += res.text
    calls.push(...res.toolCalls)
  }
  const end = splitter.flush()
  textOut += end.text
  calls.push(...end.toolCalls)
  assert.equal(textOut, "Thinking... <temp>not a tag</temp>  done.")
  assert.equal(calls.length, 1)
  assert.equal(calls[0].function.name, "read_file")
  assert.equal(calls[0].function.arguments, '{"p":1}')

  const fencedSplitter = createToolCallSplitter(true)
  const fencedOut = fencedSplitter.push('<tool_call>\n```json\n{"name":"write_file","arguments":{"path":"a.txt"}}\n```\n</tool_call>')
  const fencedEnd = fencedSplitter.flush()
  const fencedCalls = [...fencedOut.toolCalls, ...fencedEnd.toolCalls]
  assert.equal(fencedCalls.length, 1)
  assert.equal(fencedCalls[0].function.name, "write_file")
  assert.equal(fencedCalls[0].function.arguments, '{"path":"a.txt"}')
})

function makeMockWebSocketClass(responder) {
  return class MockWebSocket {
    constructor(url, init) {
      this.url = url
      this.init = init
      this.readyState = 0
      this.sent = []
      queueMicrotask(() => responder(this))
    }
    send(data) {
      this.sent.push(data)
    }
    close(code = 1000, reason = "") {
      this.readyState = 3
      this.onclose?.({ code, reason })
    }
    emitAidpEvent(obj) {
      const payload = Buffer.from(
        JSON.stringify({ payload: Buffer.from(JSON.stringify(obj), "utf8").toString("base64") }),
        "utf8",
      )
      const frame = esEncode({ ":event-type": "rawOutputEvent", ":message-type": "event" }, payload)
      this.onmessage?.({ data: frame })
    }
    emitGatewayEvent(obj) {
      const innerCbor = cborEncode({ payload: new TextEncoder().encode(JSON.stringify(obj)) })
      const frame = esEncode(
        { ":event-type": "rawOutputEvent", ":content-type": "application/cbor", ":message-type": "event" },
        innerCbor,
      )
      this.onmessage?.({ data: frame })
    }
  }
}

test("AmazonQuickAuthPlugin executes SSE streaming, 401 token retry, usage, and exports cleanly", async () => {
  let wsAttempts = 0
  const MockWS = makeMockWebSocketClass((ws) => {
    wsAttempts++
    if (ws.url.includes("expired-tok")) {
      ws.readyState = 3
      ws.onerror?.({ message: "401 Unauthorized" })
      ws.onclose?.({ code: 1008, reason: "401 Unauthorized" })
      return
    }
    const emit = (obj) =>
      ws.url.includes("ChatBidirectionalStream") ? ws.emitGatewayEvent(obj) : ws.emitAidpEvent(obj)
    ws.readyState = 1
    ws.onopen?.()
    queueMicrotask(() => {
      emit({ type: "session_established", payload: {} })
      emit({ type: "message_start", payload: {} })
      emit({ type: "tool_start", payload: { tool_name: "web_search", tool_id: "ws-1" } })
      emit({ type: "tool_result", payload: { tool_id: "ws-1", result: "ok" } })
      emit({ type: "thinking_delta", payload: { delta: "Let me think" } })
      emit({ type: "text_delta", payload: { delta: "Hello world!" } })
      emit({ type: "turn_usage", payload: { input_tokens: 10, output_tokens: 5, total_tokens: 15 } })
      emit({ type: "message_complete", payload: {} })
    })
  })

  let stored = {
    type: "oauth",
    access: "expired-tok",
    refresh: "ref-live",
    expires: Date.now() + 3600_000,
    clientId: "cid-1",
    region: "us-east-1",
    providerType: "social",
    tenantUrl: "https://qbs-test.dp.appintegrations.us-east-1.prod.plato.ai.aws.dev",
  }

  const pluginFactory = createAmazonQuickPlugin({
    WebSocket: MockWS,
    fetch: async (url) => {
      if (url.includes("oidc.us-east-1.amazonaws.com/token")) {
        return new Response(JSON.stringify({ accessToken: "fresh-tok", refreshToken: "ref-next", expiresIn: 3600 }), { status: 200 })
      }
      if (url.includes("qbs-event")) {
        return new Response(
          JSON.stringify({
            eventResponse: {
              invokeAgentResponse: JSON.stringify({
                statusCode: 200,
                body: JSON.stringify({
                  monthly: { used: 5, limit: 50, percent: 10, label: "5 / 50" },
                  subscription: { subscription_type: "PRO" },
                }),
              }),
            },
          }),
          { status: 200 },
        )
      }
      return new Response("{}", { status: 404 })
    },
  })

  const plugin = await pluginFactory({
    client: {
      auth: {
        set: async ({ body }) => {
          stored = { ...stored, ...body }
        },
      },
    },
  })

  // config hook
  const cfg = {}
  await plugin.config(cfg)
  assert.equal(cfg.provider["amazon-quick"].npm, "@ai-sdk/openai-compatible")
  assert.equal(cfg.provider["amazon-quick"].api, "https://quick.aws.com/v1")
  assert.equal(cfg.provider["amazon-quick"].models.balanced.api.url, "https://quick.aws.com/v1")

  // loader SSE stream with 401 refresh recovery (plus Uint8Array body & non-empty baseURL)
  const loader = await plugin.auth.loader(async () => stored)
  assert.equal(loader.baseURL, "https://qbs-test.dp.appintegrations.us-east-1.prod.plato.ai.aws.dev")
  const res = await loader.fetch(`${loader.baseURL}/chat/completions`, {
    method: "POST",
    body: new TextEncoder().encode(JSON.stringify({ model: "balanced@high", stream: true, messages: [{ role: "user", content: "Hi" }] })),
  })
  assert.equal(res.status, 200)
  assert.equal(res.headers.get("X-Magpie-Sign-In"), "renewed")
  assert.equal(wsAttempts, 2)
  const sseText = await res.text()
  assert.ok(sseText.includes('"reasoning_content":"Let me think"'))
  assert.ok(sseText.includes('"content":"Hello world!"'))
  assert.ok(sseText.includes('"finish_reason":"stop"'))
  assert.ok(sseText.includes("data: [DONE]"))

  // Non-streaming Gateway path
  const gwRes = await executeSncsChat(
    { providerType: "gateway", access: "tok-ok", tenantUrl: "https://gw.desktop.enterprise.quick.aws.com" },
    { model: "fast", stream: false, messages: [{ role: "user", content: "Ping" }] },
    { WebSocket: MockWS },
  )
  const gwJson = await gwRes.json()
  assert.equal(gwJson.choices[0].message.content, "Hello world!")
  assert.equal(gwJson.usage.total_tokens, 15)

  // Usage hook
  const usageRes = await plugin.auth.usage(async () => stored)
  assert.equal(usageRes.plan, "Pro")
  assert.equal(usageRes.windows[0].name, "Monthly usage")
  assert.equal(usageRes.signIn, "kept")

  // opencode1 & opencode2 entrypoint sanity
  assert.equal(typeof opencode1.AmazonQuickAuthPlugin, "function")
  assert.equal(typeof opencode2.id, "string")
  assert.equal(typeof opencode2.setup, "function")
  const addedMethods = []
  const addedProviders = []
  await opencode2.setup({
    integration: {
      transform: async (fn) =>
        fn({
          update: () => {},
          method: { update: (x) => addedMethods.push(x) },
        }),
      connection: { active: async () => undefined },
    },
    provider: {
      transform: async (fn) => fn({ add: (x) => addedProviders.push(x) }),
      reload: async () => {},
    },
    aisdk: { hook: async () => {} },
  })
  assert.equal(addedMethods.length, 3)
  assert.equal(addedMethods[0].integrationID, "amazon-quick")
  assert.equal(addedProviders[0].info.id, "amazon-quick")

  const pkgJson = JSON.parse(readFileSync(new URL("./package.json", import.meta.url), "utf8"))
  assert.ok(pkgJson.magpie?.icon?.startsWith("data:image/png;base64,"))
  const decodedIcon = Buffer.from(pkgJson.magpie.icon.slice("data:image/png;base64,".length), "base64")
  assert.equal(decodedIcon.subarray(0, 4).toString("hex"), "89504e47")

  // Unrecoverable 401 behavior: desktop stays "kept", OAuth marks "expired"
  const RejectWS = makeMockWebSocketClass((ws) => {
    ws.readyState = 3
    ws.onerror?.({ message: "401 Unauthorized" })
    ws.onclose?.({ code: 1008, reason: "401 Unauthorized" })
  })
  const rejectFactory = createAmazonQuickPlugin({
    WebSocket: RejectWS,
    fetch: async () => new Response(JSON.stringify({ error: "invalid_grant" }), { status: 400 }),
  })
  const rejectPlugin = await rejectFactory({})
  const oauthLoader = await rejectPlugin.auth.loader(async () => ({
    type: "oauth",
    access: "bad-tok",
    refresh: "bad-ref",
    clientId: "cid-1",
    region: "us-east-1",
    providerType: "social",
    tenantUrl: "https://qbs-test.dp.appintegrations.us-east-1.prod.plato.ai.aws.dev",
  }))
  const oauth401 = await oauthLoader.fetch("https://unused.local/chat/completions", {
    method: "POST",
    body: JSON.stringify({ model: "fast", messages: [{ role: "user", content: "Hi" }] }),
  })
  assert.equal(oauth401.status, 401)
  assert.equal(oauth401.headers.get("X-Magpie-Sign-In"), "expired")

  const dir401 = mkdtempSync(join(tmpdir(), "aq-401-"))
  try {
    writeFileSync(
      join(dir401, "profiles.json"),
      JSON.stringify({
        version: 1,
        last_active: "p1",
        entries: [{ id: "p1", provider: "social", login_metadata: { tenant_url: "https://qbs.dp.appintegrations.us-east-1.prod.plato.ai.aws.dev" } }],
      }),
    )
    const desktopRejectFactory = createAmazonQuickPlugin({
      dir: dir401,
      WebSocket: RejectWS,
      readKeyring: () =>
        JSON.stringify({
          access_token: "desk-bad",
          tenant_url: "https://qbs.dp.appintegrations.us-east-1.prod.plato.ai.aws.dev",
        }),
    })
    const desktopRejectPlugin = await desktopRejectFactory({})
    const deskLoader = await desktopRejectPlugin.auth.loader(async () => ({ type: "oauth", source: "desktop", profileId: "p1" }))
    const desk401 = await deskLoader.fetch("https://unused.local/chat/completions", {
      method: "POST",
      body: JSON.stringify({ model: "fast", messages: [{ role: "user", content: "Hi" }] }),
    })
    assert.equal(desk401.status, 401)
    assert.equal(desk401.headers.get("X-Magpie-Sign-In"), "kept")
  } finally {
    rmSync(dir401, { recursive: true, force: true })
  }
})

test("browserSignIn registers quicksight:read_write OIDC client and handles loopback callback", async () => {
  const dir = mkdtempSync(join(tmpdir(), "aq-browser-"))
  try {
    writeFileSync(
      join(dir, "profiles.json"),
      JSON.stringify({
        version: 1,
        last_active: "p1",
        entries: [
          {
            id: "p1",
            provider: "social",
            login_metadata: {
              issuer_url: "https://identitycenter.amazonaws.com/ssoins-7223e642b2afb64c",
              tenant_url: "https://qbs-test.dp.appintegrations.us-east-1.prod.plato.ai.aws.dev",
            },
          },
        ],
      }),
    )

    let registeredBody = null
    let tokenBody = null
    const mockFetch = async (url, init) => {
      if (url.endsWith("/client/register")) {
        registeredBody = JSON.parse(init.body)
        return new Response(JSON.stringify({ clientId: "cid-pub", clientSecret: "csec-1" }), { status: 200 })
      }
      if (url.endsWith("/token")) {
        tokenBody = JSON.parse(init.body)
        return new Response(JSON.stringify({ accessToken: "tok-browser", refreshToken: "ref-browser", expiresIn: 3600 }), { status: 200 })
      }
      return new Response("{}", { status: 404 })
    }

    const flow = await browserSignIn({}, { dir, fetch: mockFetch })
    assert.deepEqual(registeredBody.scopes, ["quicksight:read_write"])
    assert.ok(registeredBody.redirectUris[0].startsWith("http://127.0.0.1:"))

    const u = new URL(flow.url)
    assert.equal(u.origin, "https://oidc.us-east-1.amazonaws.com")
    assert.equal(u.searchParams.get("scope"), "quicksight:read_write")
    assert.equal(u.searchParams.get("prompt"), "login")
    const redirectUri = u.searchParams.get("redirect_uri")
    const state = u.searchParams.get("state")

    const cbPromise = flow.callback()
    const hit = await fetch(`${redirectUri}/?code=auth-code-99&state=${encodeURIComponent(state)}`)
    assert.equal(hit.status, 200)

    const result = await cbPromise
    assert.equal(result.type, "success")
    assert.equal(result.access, "tok-browser")
    assert.equal(result.refresh, "ref-browser")
    assert.equal(result.providerType, "social")
    assert.equal(result.tenantUrl, "https://qbs-test.dp.appintegrations.us-east-1.prod.plato.ai.aws.dev")
    assert.equal(tokenBody.code, "auth-code-99")
    assert.ok(tokenBody.codeVerifier)
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})
