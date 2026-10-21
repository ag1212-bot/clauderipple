// An `openai-compatible` provider may hold several credentials (`providers.<name>.credentials`), and
// a refused key is replaced inside the same turn — before the first byte reaches the client — the way
// the anthropic-compatible wire has always worked. These drive the real proxy and real HTTP fakes;
// the keys are fake. A 400 is not walked across the pool, a 401 quarantines its key, and each key of
// an OpenCode Go provider reports its own usage.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import http from "node:http";
import net from "node:net";
import os from "node:os";
import path from "node:path";
import tls from "node:tls";

import { CertStore } from "../src/certs.ts";
import { DEFAULTS, type Config, type Provider } from "../src/config.ts";
import { UpstreamHealth } from "../src/health.ts";
import { Logger } from "../src/log.ts";
import { Proxy } from "../src/proxy.ts";
import { RequestLog } from "../src/requestlog.ts";
import { createCa } from "../src/x509.ts";

function freePort(): Promise<number> {
  return new Promise((resolve) => {
    const probe = net.createServer();
    probe.listen(0, "127.0.0.1", () => {
      const { port } = probe.address() as net.AddressInfo;
      probe.close(() => resolve(port));
    });
  });
}

type Reply = { status: number; headers?: Record<string, string>; body?: string };
type Upstream = {
  port: number;
  keys: string[];
  /** Every request's headers, in order, for the ones a test reads more than the key from. */
  headers: http.IncomingHttpHeaders[];
  /** GET /usage answers: which key asked, and how many times. */
  usageSeen: { key: string; at: number }[];
  /** What the usage endpoint answers per key: rolling percent, and where that window resets. */
  usage: Map<string, { percent: number; resetsAt: string }>;
  close: () => Promise<void>;
};

/**
 * A Chat Completions endpoint that answers `reply` per request and records the key each one carried.
 * A GET to `/usage` is the OpenCode Go usage route: it answers per key from `usage`, and a key with
 * no entry answers 40% used, resetting in an hour.
 */
async function upstream(reply: (key: string, n: number) => Reply): Promise<Upstream> {
  const keys: string[] = [];
  const headers: http.IncomingHttpHeaders[] = [];
  const usageSeen: { key: string; at: number }[] = [];
  const usage = new Map<string, { percent: number; resetsAt: string }>();
  const server = http.createServer((req, res) => {
    const key = String(req.headers.authorization ?? "");
    if (req.method === "GET" && (req.url ?? "").endsWith("/usage")) {
      usageSeen.push({ key, at: Date.now() });
      const row = usage.get(key) ?? { percent: 40, resetsAt: new Date(Date.now() + 3_600_000).toISOString() };
      res.writeHead(200, { "content-type": "application/json" });
      res.end(JSON.stringify({ usage: {
        rolling: { status: "ok", percent: row.percent, resetsAt: row.resetsAt },
        weekly: { status: "ok", percent: 10, resetsAt: new Date(Date.now() + 3 * 86_400_000).toISOString() },
        monthly: { status: "ok", percent: 5, resetsAt: new Date(Date.now() + 20 * 86_400_000).toISOString() },
      } }));
      return;
    }
    const chunks: Buffer[] = [];
    req.on("data", (chunk: Buffer) => chunks.push(chunk));
    req.on("end", () => {
      keys.push(key);
      headers.push(req.headers);
      const r = reply(key, keys.length);
      res.writeHead(r.status, { "content-type": "application/json", ...(r.headers ?? {}) });
      res.end(r.body ?? "{}");
    });
  });
  await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
  return {
    port: (server.address() as net.AddressInfo).port,
    keys,
    headers,
    usageSeen,
    usage,
    close: () => new Promise<void>((r) => server.close(() => r())),
  };
}

/** What a Chat Completions vendor answers to a turn it accepts: a stream, as the adapter asks for. */
const answer = (text: string): Reply => ({
  status: 200,
  headers: { "content-type": "text/event-stream" },
  body: [
    { choices: [{ delta: { content: text }, finish_reason: null }] },
    { choices: [{ delta: {}, finish_reason: "stop" }] },
    { choices: [], usage: { prompt_tokens: 5, completion_tokens: 2 } },
  ].map((record) => `data: ${JSON.stringify(record)}\n\n`).join("") + "data: [DONE]\n\n",
});

const rateLimited: Reply = { status: 429, headers: { "retry-after": "300" }, body: JSON.stringify({ error: { message: "slow down" } }) };

type Rig = {
  proxy: Proxy;
  send: (conversation?: string) => Promise<number>;
  /** The same request as `send`, but `/v1/messages/count_tokens`, which the router answers locally. */
  sendCount: (conversation?: string) => Promise<number>;
  stop: () => Promise<void>;
};

/** A live proxy in front of `providers`, reached the way Claude Code reaches it. */
async function rig(providers: Config["providers"], routes: Config["routes"], opts: { usageFetch?: (url: string, init?: RequestInit) => Promise<Response>; usageTtlMs?: number } = {}): Promise<Rig> {
  const home = fs.mkdtempSync(path.join(os.tmpdir(), "cr-openai-keys-"));
  const ca = createCa({ cn: "clauderipple test" });
  fs.writeFileSync(path.join(home, "ca.pem"), ca.certPem);
  fs.writeFileSync(path.join(home, "ca.key"), ca.keyPem);
  const proxyPort = await freePort();
  const cfg: Config = { ...DEFAULTS, listen: { host: "127.0.0.1", port: proxyPort }, providers, routes, direct: [] };
  const proxy = new Proxy({
    config: () => cfg,
    log: new Logger(path.join(home, "router.log"), 1e6, 1, false),
    certs: new CertStore(home),
    health: new UpstreamHealth(() => 100, () => {}),
    home,
    requests: new RequestLog(path.join(home, "requests.jsonl")),
    ...(opts.usageFetch || opts.usageTtlMs !== undefined
      ? { opencodeUsage: { ...(opts.usageFetch ? { fetch: opts.usageFetch } : {}), ...(opts.usageTtlMs !== undefined ? { ttlMs: opts.usageTtlMs } : {}) } }
      : {}),
  });
  await proxy.listen();

  const send = async (conversation = "user-1", path = "/v1/messages"): Promise<number> => {
    const sock = net.connect(proxyPort, "127.0.0.1");
    await new Promise<void>((r) => sock.once("connect", () => r()));
    sock.write("CONNECT api.anthropic.com:443 HTTP/1.1\r\nHost: api.anthropic.com:443\r\n\r\n");
    await new Promise<void>((r) => sock.once("data", () => r()));
    const secure = tls.connect({ socket: sock, servername: "api.anthropic.com", ca: ca.certPem });
    await new Promise<void>((r) => secure.once("secureConnect", () => r()));
    const body = JSON.stringify({
      model: "claude-opus-4-8",
      max_tokens: 16,
      stream: false,
      metadata: { user_id: conversation },
      messages: [{ role: "user", content: "hi" }],
    });
    secure.write(`POST ${path} HTTP/1.1\r\nHost: api.anthropic.com\r\ncontent-type: application/json\r\ncontent-length: ${Buffer.byteLength(body)}\r\n\r\n${body}`);
    const first = await new Promise<Buffer>((r) => secure.once("data", (d: Buffer) => setTimeout(() => r(d), 80)));
    secure.destroy();
    return Number(/^HTTP\/1\.1 (\d{3})/.exec(first.toString("latin1"))?.[1] ?? 0);
  };

  return { proxy, send, sendCount: (conversation = "user-1") => send(conversation, "/v1/messages/count_tokens"), stop: async () => { proxy.close(); fs.rmSync(home, { recursive: true, force: true }); } };
}

/** Two keys for one provider; each is a complete header set, as the config declares them. */
const pooled = (port: number, extra: Partial<Provider> = {}): Provider => ({
  type: "openai-compatible",
  url: `http://127.0.0.1:${port}/v1`,
  credentials: [
    { id: "key-1", label: "First key", headers: { authorization: "Bearer test-key-1" } },
    { id: "key-2", label: "Second key", headers: { authorization: "Bearer test-key-2" } },
  ],
  ...extra,
} as Provider);

/** The same two keys on a provider the router recognises as OpenCode Go (its usage route exists). */
const goPooled = (port: number, extra: Partial<Provider> = {}): Provider =>
  pooled(port, { preset: "opencode-go", ...extra });

const routes: Config["routes"] = { "claude-opus-4-8": { provider: "p", model: "vendor-model" } };

test("a rate-limited key hands the same turn to the next key, and the client sees one 200", async () => {
  const up = await upstream((key) => key === "Bearer test-key-1" ? rateLimited : answer("second key"));
  const r = await rig({ p: pooled(up.port) }, routes);
  try {
    assert.equal(await r.send(), 200, "answered on the first ask");
    assert.deepEqual(up.keys, ["Bearer test-key-1", "Bearer test-key-2"], "both keys tried, within one turn");
  } finally { await r.stop(); await up.close(); }
});

test("the conversation stays on the key that answered while the limited one cools", async () => {
  // The first key is limited for the whole test. Turn 1 discovers it and lands on key 2; turn 2 must
  // start there — moving it back would take the prompt cache with it (§4).
  const up = await upstream((key) => key === "Bearer test-key-1" ? rateLimited : answer("second key"));
  const r = await rig({ p: pooled(up.port) }, routes);
  try {
    assert.equal(await r.send(), 200);
    assert.equal(await r.send(), 200);
    assert.deepEqual(up.keys, ["Bearer test-key-1", "Bearer test-key-2", "Bearer test-key-2"], "turn 2 starts on the key that answered");
  } finally { await r.stop(); await up.close(); }
});

test("a 400 is the request's own fault: it is not retried against the other key", async () => {
  const up = await upstream(() => ({ status: 400, body: JSON.stringify({ error: { message: "bad body" } }) }));
  const r = await rig({ p: pooled(up.port) }, routes);
  try {
    assert.equal(await r.send(), 400);
    assert.equal(up.keys.length, 1, "asked once, not walked across the pool");
  } finally { await r.stop(); await up.close(); }
});

test("a 401 quarantines the key that produced it; the next turn uses the other one", async () => {
  const up = await upstream((key) => key === "Bearer test-key-1" ? { status: 401, body: JSON.stringify({ error: { message: "invalid key" } }) } : answer("second key"));
  const r = await rig({ p: pooled(up.port) }, routes);
  try {
    assert.equal(await r.send(), 200);
    // Reported for the dashboard: key-1 is quarantined, key-2 is ready.
    const report = r.proxy.credentialHealth().p ?? [];
    assert.deepEqual(report.map((entry) => [entry.id, entry.state]), [["key-1", "quarantined"], ["key-2", "ready"]]);
    assert.equal(await r.send(), 200);
    assert.equal(up.keys.at(-1), "Bearer test-key-2", "a quarantined key is not asked again");
  } finally { await r.stop(); await up.close(); }
});

test("OpenCode Go reports one limits entry per key, read with that key's own headers", async () => {
  // The `/usage` endpoint answers a different percentage per key, so a lookup that ignored the key
  // would show the same numbers under both entries.
  const percentFor: Record<string, number> = { "Bearer test-key-1": 11, "Bearer test-key-2": 77 };
  const usageSeen: string[] = [];
  const usage = http.createServer((req, res) => {
    const auth = String(req.headers.authorization ?? "");
    usageSeen.push(auth);
    res.writeHead(200, { "content-type": "application/json" });
    res.end(JSON.stringify({ usage: {
      rolling: { status: "ok", percent: percentFor[auth] ?? 0, resetsAt: "2026-10-21T05:39:48Z" },
      weekly: { status: "ok", percent: 52, resetsAt: "2026-10-25T00:00:00Z" },
      monthly: { status: "ok", percent: 92, resetsAt: "2026-11-20T00:00:00Z" },
    } }));
  });
  await new Promise<void>((r) => usage.listen(0, "127.0.0.1", r));
  const port = (usage.address() as net.AddressInfo).port;
  const provider: Provider = { ...pooled(port), preset: "opencode-go" } as Provider;
  const r = await rig({ p: provider }, routes);
  try {
    const rows = await r.proxy.opencodeLimits("p");
    assert.deepEqual(rows.map((row) => [row.id, row.label]), [["key-1", "First key"], ["key-2", "Second key"]]);
    assert.deepEqual(rows.map((row) => row.windows.find((w) => w.kind === "five_hour")?.used_percent), [11, 77]);
    assert.deepEqual([...usageSeen].sort(), ["Bearer test-key-1", "Bearer test-key-2"]);
  } finally { await r.stop(); await new Promise<void>((done) => usage.close(() => done())); }
});

test("a provider with no credentials still reports its one limits entry", async () => {
  const usage = http.createServer((_req, res) => {
    res.writeHead(200, { "content-type": "application/json" });
    res.end(JSON.stringify({ usage: { rolling: { status: "ok", percent: 5, resetsAt: "2026-10-21T05:39:48Z" } } }));
  });
  await new Promise<void>((r) => usage.listen(0, "127.0.0.1", r));
  const port = (usage.address() as net.AddressInfo).port;
  const r = await rig({ p: { type: "openai-compatible", preset: "opencode-go", url: `http://127.0.0.1:${port}/v1`, headers: { authorization: "Bearer only-key" } } as Provider }, routes);
  try {
    const rows = await r.proxy.opencodeLimits("p");
    assert.equal(rows.length, 1);
    assert.equal(rows[0]!.id, "default");
    assert.equal(rows[0]!.windows[0]!.used_percent, 5);
  } finally { await r.stop(); await new Promise<void>((done) => usage.close(() => done())); }
});

test("a key whose 5-hour window is full is skipped before the turn, and a new conversation starts on the next key", async () => {
  // key-1 has spent its 5-hour window; the account card knows it before any request is made here.
  const up = await upstream(() => answer("second key"));
  const reset = new Date(Date.now() + 2 * 3_600_000).toISOString();
  up.usage.set("Bearer test-key-1", { percent: 100, resetsAt: reset });
  const r = await rig({ p: goPooled(up.port) }, routes);
  try {
    // Reading the cards is what fills the usage cache the router reads (the real one fills it on a
    // timer); the turn itself only ever looks at that cache.
    const accounts = await r.proxy.opencodeAccounts("p");
    assert.deepEqual(accounts.map((a) => [a.id, a.state]), [["key-1", "cooling"], ["key-2", "in-use"]]);
    assert.equal(accounts[0]!.cooldown_seconds! > 3_000, true, "cooling until that window resets");

    assert.equal(await r.send(), 200);
    assert.deepEqual(up.keys, ["Bearer test-key-2"], "the spent key was never asked");
    // And the other conversation does not drag it back either.
    assert.equal(await r.send("user-2"), 200);
    assert.deepEqual(up.keys, ["Bearer test-key-2", "Bearer test-key-2"]);
  } finally { await r.stop(); await up.close(); }
});

test("a charged 429 cools its key until the reset the vendor states in the body", async () => {
  // key-1 refuses with OpenCode's usage-limit shape: 100% of the weekly window, and when it resets.
  const resetsAt = new Date(Date.now() + 90 * 60_000).toISOString();
  const up = await upstream((key) => key === "Bearer test-key-1"
    ? {
      status: 429,
      body: JSON.stringify({ type: "error", error: { type: "rate_limit_error", message: "Weekly limit reached", usage: { weekly: { percent: 100, resetsAt } } } }),
    }
    : answer("second key"));
  const r = await rig({ p: goPooled(up.port) }, routes);
  try {
    assert.equal(await r.send(), 200);
    assert.deepEqual(up.keys, ["Bearer test-key-1", "Bearer test-key-2"], "the same turn moved on before the first byte");
    const accounts = await r.proxy.opencodeAccounts("p");
    // ~90 minutes of cooldown, from the body's own reset — not the pool's one-minute default.
    const cooldown = accounts[0]!.cooldown_seconds ?? 0;
    assert.equal(cooldown > 5_000 && cooldown <= 5_400, true, `cooling until the stated reset, got ${cooldown}s`);
    assert.equal(accounts[1]!.state, "in-use");
  } finally { await r.stop(); await up.close(); }
});

test("a 429 that states no reset is cooled to the full window's reset, found by one usage read", async () => {
  const up = await upstream((key) => key === "Bearer test-key-1"
    ? { status: 429, body: JSON.stringify({ error: { message: "slow down" } }) }
    : answer("second key"));
  const r = await rig({ p: goPooled(up.port) }, routes);
  try {
    // The cache first says key-1 is fine, so a key cooled from the cache alone would be at a minute.
    await r.proxy.opencodeAccounts("p");
    // What the fresh read finds: the 5-hour window is full and resets in three hours.
    const reset = new Date(Date.now() + 3 * 3_600_000).toISOString();
    up.usage.set("Bearer test-key-1", { percent: 100, resetsAt: reset });

    assert.equal(await r.send(), 200);
    assert.deepEqual(up.keys, ["Bearer test-key-1", "Bearer test-key-2"]);
    // The refusal made the router read that key's usage once more, past the cache.
    assert.equal(up.usageSeen.filter((seen) => seen.key === "Bearer test-key-1").length >= 2, true, "usage was read again");
    const accounts = await r.proxy.opencodeAccounts("p");
    assert.equal(accounts[0]!.state, "cooling");
    assert.equal((accounts[0]!.cooldown_seconds ?? 0) > 3_000, true, "cooled past a fallback minute, to the window's reset");
  } finally { await r.stop(); await up.close(); }
});

test("a 429 with nothing stated and no full window falls back to one minute", async () => {
  const up = await upstream((key) => key === "Bearer test-key-1" ? { status: 429, body: "{}" } : answer("second key"));
  const r = await rig({ p: goPooled(up.port) }, routes);
  try {
    assert.equal(await r.send(), 200);
    const accounts = await r.proxy.opencodeAccounts("p");
    const cooldown = accounts[0]!.cooldown_seconds ?? 0;
    assert.equal(cooldown > 0 && cooldown <= 60, true, `a minute by default, got ${cooldown}s`);
  } finally { await r.stop(); await up.close(); }
});

test("a paused key is never used, and the turn goes to the next one", async () => {
  const up = await upstream(() => answer("second key"));
  const provider = goPooled(up.port);
  (provider as { credentials?: { paused?: boolean }[] }).credentials![0]!.paused = true;
  const r = await rig({ p: provider }, routes);
  try {
    assert.equal(await r.send(), 200);
    assert.deepEqual(up.keys, ["Bearer test-key-2"], "a paused key is not asked at all");
    const accounts = await r.proxy.opencodeAccounts("p");
    assert.deepEqual(accounts.map((a) => a.state), ["paused", "in-use"]);
  } finally { await r.stop(); await up.close(); }
});

test("with one key paused and the other cooling, the turn still goes out on the cooling one", async () => {
  // The all-paused 503 does not cover this: key-2 is out of rotation only until its cooldown ends, so
  // there is still a key to send on. The fallback used to be `credentials[0]`, which here is the
  // paused key — the exact key the operator took out of rotation.
  const up = await upstream((_key, n) => n === 1 ? rateLimited : answer("second key"));
  const provider = goPooled(up.port);
  (provider as { credentials?: { paused?: boolean }[] }).credentials![0]!.paused = true;
  const r = await rig({ p: provider }, routes);
  try {
    assert.equal(await r.send(), 429, "the cooling key's own refusal reaches the client once");
    // Turn 2: key-1 paused, key-2 cooling, so `pick` has nothing. The turn must still carry key-2.
    assert.equal(await r.send(), 200);
    assert.deepEqual(up.keys, ["Bearer test-key-2", "Bearer test-key-2"], "the paused key is never asked, even as the last resort");
  } finally { await r.stop(); await up.close(); }
});

test("every key paused is answered locally instead of sending the turn on the first one", async () => {
  // Pausing is the operator's decision, not a failure: there is no key left to send the turn on, and
  // falling back to key 1 would use a key the dashboard says is out of rotation.
  const up = await upstream(() => answer("should not be called"));
  const provider = goPooled(up.port);
  for (const credential of (provider as { credentials: { paused?: boolean }[] }).credentials) credential.paused = true;
  const r = await rig({ p: provider }, routes);
  try {
    assert.equal(await r.send(), 503);
    assert.deepEqual(up.keys, [], "nothing went upstream");
    const accounts = await r.proxy.opencodeAccounts("p");
    assert.deepEqual(accounts.map((a) => a.state), ["paused", "paused"]);
  } finally { await r.stop(); await up.close(); }
});

test("a 429's retry-after is the cooldown, not the pool's one-minute default", async () => {
  // A relay answering `retry-after: 3600` means an hour. Judged without it, that key was retried after
  // a minute and refused again, once a minute for the whole hour.
  const up = await upstream((key) => key === "Bearer test-key-1"
    ? { status: 429, headers: { "retry-after": "3600" }, body: JSON.stringify({ error: { message: "slow down" } }) }
    : answer("second key"));
  const r = await rig({ p: pooled(up.port) }, routes);
  try {
    assert.equal(await r.send(), 200);
    const report = r.proxy.credentialHealth().p ?? [];
    const cooling = report.find((entry) => entry.id === "key-1")!.cooldownSeconds ?? 0;
    assert.equal(cooling > 3_500 && cooling <= 3_600, true, `an hour, got ${cooling}s`);
  } finally { await r.stop(); await up.close(); }
});

test("a relay's 403 about its own broken upstream does not cool the key that carried it", async () => {
  // "Upstream request failed" is the relay saying its own call broke, not a statement about our key.
  // Charged without the body it read as an auth failure and parked the key for a minute, which moved
  // the conversation off it and took the prompt cache with it (ARCHITECTURE 2026-09-21).
  const relay = { status: 403, body: JSON.stringify({ error: { message: "Error from provider (Console Go): Upstream request failed" } }) };
  const up = await upstream(() => relay);
  const r = await rig({ p: pooled(up.port) }, routes);
  try {
    assert.equal(await r.send(), 403);
    const report = r.proxy.credentialHealth().p ?? [];
    assert.deepEqual(report.map((entry) => [entry.id, entry.state]), [["key-1", "ready"], ["key-2", "ready"]]);
  } finally { await r.stop(); await up.close(); }
});

test("a local count_tokens answer charges nobody, so it cannot lift a quarantine it did not earn", async () => {
  const up = await upstream((key) => key === "Bearer test-key-1"
    ? { status: 401, body: JSON.stringify({ error: { message: "invalid key" } }) }
    : answer("second key"));
  const r = await rig({ p: pooled(up.port) }, routes);
  try {
    assert.equal(await r.send(), 200);
    assert.equal((r.proxy.credentialHealth().p ?? []).find((entry) => entry.id === "key-1")?.state, "quarantined");
    assert.equal(await r.sendCount("user-1"), 200);
    assert.equal(
      (r.proxy.credentialHealth().p ?? []).find((entry) => entry.id === "key-1")?.state,
      "quarantined",
      "count_tokens is answered locally; a local 200 must not clear a quarantine",
    );
  } finally { await r.stop(); await up.close(); }
});

test("when the cooldown expires, a new conversation returns to the first key", async () => {
  // List order is the priority, so a key that came back is key 1 again without anything having to
  // move a conversation: only the conversation already on key 2 stays there.
  const up = await upstream((key, n) => key === "Bearer test-key-1" && n === 1
    ? { status: 429, headers: { "retry-after": "1" }, body: JSON.stringify({ error: { message: "slow down" } }) }
    : answer("second key"));
  const r = await rig({ p: pooled(up.port) }, routes);
  try {
    assert.equal(await r.send(), 200);
    assert.equal(up.keys.at(-1), "Bearer test-key-2");
    await new Promise((resolve) => setTimeout(resolve, 1_200));
    // The old conversation keeps the key that answered it (its prompt cache lives there).
    assert.equal(await r.send("user-1"), 200);
    assert.equal(up.keys.at(-1), "Bearer test-key-2", "a running conversation stays where its cache is");
    assert.equal(await r.send("user-2"), 200);
    assert.equal(up.keys.at(-1), "Bearer test-key-1", "a new conversation starts on the key that is back");
  } finally { await r.stop(); await up.close(); }
});

test("the provider's own headers still travel with a credential; only its auth is replaced", async () => {
  // An organization id, a beta flag or a custom header used to disappear the moment the provider had
  // credentials configured, because the credential's headers replaced the provider's outright.
  const up = await upstream(() => answer("second key"));
  const provider = pooled(up.port, { headers: { authorization: "Bearer stale", "x-org": "acme" } });
  const r = await rig({ p: provider }, routes);
  try {
    assert.equal(await r.send(), 200);
    const sent = up.headers.at(-1)!;
    assert.equal(sent.authorization, "Bearer test-key-1", "the key in rotation, not the configured one");
    assert.equal(sent["x-org"], "acme", "the provider's own header came along");
  } finally { await r.stop(); await up.close(); }
});

test("a key whose usage has never been read reports unknown, and a failed read keeps the last good answer", async () => {
  // The account card used to derive "in-use" from a live read: a failed read left no windows, and the
  // card then presented a possibly spent key as the one in use. The read is injected here so no test
  // ever reaches the vendor.
  const up = await upstream(() => answer("second key"));
  let failing = true;
  const fetched: string[] = [];
  const usageFetch = async (url: string, init?: RequestInit): Promise<Response> => {
    fetched.push(String((init?.headers as Record<string, string>)?.authorization ?? url));
    if (failing) return new Response("nope", { status: 500 });
    return new Response(JSON.stringify({ usage: {
      rolling: { status: "ok", percent: 12, resetsAt: new Date(Date.now() + 3_600_000).toISOString() },
      weekly: { status: "ok", percent: 20, resetsAt: new Date(Date.now() + 3 * 86_400_000).toISOString() },
    } }), { status: 200 });
  };
  const r = await rig({ p: goPooled(up.port) }, routes, { usageFetch, usageTtlMs: 0 });
  try {
    const unread = await r.proxy.opencodeAccounts("p");
    assert.deepEqual(unread.map((a) => [a.id, a.state, a.windows.length, a.at]), [
      ["key-1", "unknown", 0, 0],
      ["key-2", "unknown", 0, 0],
    ]);
    assert.equal(fetched.length >= 2, true, "the dashboard did ask (and the fake answered 500)");

    failing = false;
    const read = await r.proxy.opencodeAccounts("p");
    assert.deepEqual(read.map((a) => [a.state, a.windows.length]), [["in-use", 2], ["standby", 2]]);
    assert.equal(read[0]!.at > 0, true);

    // And a read that fails afterwards keeps those windows instead of blanking them.
    failing = true;
    const kept = await r.proxy.opencodeAccounts("p");
    assert.deepEqual(kept.map((a) => a.windows.length), [2, 2], "the last good answer survives a failed read");
    assert.equal(kept[0]!.state, "in-use");
  } finally { await r.stop(); await up.close(); }
});
