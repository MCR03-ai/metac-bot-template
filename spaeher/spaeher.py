# KI-Spaeher: 7 KIs ueber OpenRouter, Links selbst pruefen. Kostenbremse MIN_REST_USD.
import concurrent.futures as cf
import datetime as dt
import json
import os
import re
import sys
import urllib.error
import urllib.request

API = "https://openrouter.ai/api/v1"
KEY = os.environ["OPENROUTER_API_KEY"]
MIN_REST_USD = float(os.environ.get("MIN_REST_USD", "5"))
JUDGE_PROMPT = open("spaeher/judge.txt", encoding="utf-8").read()
PROMPT = open(os.environ.get("PROMPT_FILE", "spaeher/prompt.txt"), encoding="utf-8").read()

# Panel (25.11.: Grok 2,16 USD/Lauf zu teuer, Gemini oft 429 -> entfernt). Ziel ca. 1 USD/Lauf.
PANEL = [
  {"model": "perplexity/sonar-deep-research", "web": False},
  {"model": "deepseek/deepseek-v4.1-flash", "web": True},
  {"model": "qwen/qwen4-max", "web": True},
  {"model": "z-ai/glm-5.3", "web": True},
  {"model": "moonshotai/kimi-k3.1-pro", "web": True},
]
JUDGE = "anthropic/claude-sonnet-5.5"


def call(path, body=None, timeout=900):
  req = urllib.request.Request(
    API + path,
    data=json.dumps(body).encode() if body is not None else None,
    headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
  )
  with urllib.request.urlopen(req, timeout=timeout) as r:
    return json.loads(r.read())


def rest_guthaben():
  d = call("/credits")["data"]
  return float(d["total_credits"]) - float(d["total_usage"])


def key_info():
  try:
    d = call("/key")["data"]
    return f"Schluessel-Limit: {d.get('limit')}, verbraucht: {d.get('usage')}, Rest: {d.get('limit_remaining')}"
  except Exception as e:
    return f"Schluessel-Info nicht lesbar: {e}"


def fehlertext(e):
  if isinstance(e, urllib.error.HTTPError):
    try:
      return f"HTTP {e.code}: {e.read().decode('utf-8','ignore')[:400]}"
    except Exception:
      return f"HTTP {e.code}"
  return str(e)[:400]


def frage(entry):
  body = {
    "model": entry["model"],
    "messages": [{"role": "user", "content": PROMPT}],
    "reasoning": {"effort": "high"},
    "max_tokens": 12000,
    "usage": {"include": True},
  }
  if entry["web"]:
    body["plugins"] = [{"id": "web", "max_results": 10}]
  try:
    r = call("/chat/completions", body)
    text = r["choices"][0]["message"]["content"] or ""
    cost = (r.get("usage") or {}).get("cost", 0) or 0
    return {"model": entry["model"], "text": text, "cost": cost}
  except Exception as e:
    return {"model": entry["model"], "text": "", "cost": 0, "error": fehlertext(e)}


def seite_holen(url):
  try:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=25) as r:
      html = r.read(600_000).decode("utf-8", "ignore")
    text = re.sub(r"(?s)<(script|style).*?</\1>", " ", html)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text)[:15000]
  except Exception as e:
    return f"FEHLER BEIM ABRUF: {e}"


def main():
  rest = rest_guthaben()
  print(f"Restguthaben: {rest:.2f} USD | {key_info()}", flush=True)
  if rest < MIN_REST_USD:
    sys.exit(f"Abbruch: Restguthaben {rest:.2f} < {MIN_REST_USD} USD (Puffer fuer Metaculus)")

  with cf.ThreadPoolExecutor(len(PANEL)) as ex:
    antworten = list(ex.map(frage, PANEL))
  for a in antworten:
    print(f"{a['model']}: {len(a['text'])} Zeichen, {a['cost']:.3f} USD {('FEHLER ' + a['error']) if a.get('error') else ''}", flush=True)

  stamp = dt.datetime.utcnow().strftime("%Y-%m-%d_%H%M")
  os.makedirs("out", exist_ok=True)
  with open(f"out/rohantworten_{stamp}.md", "w", encoding="utf-8") as f:
    for a in antworten:
      f.write(f"\n---\n## Rohantwort {a['model']} ({a['cost']:.3f} USD)\n\n")
      f.write(a.get("error") and f"FEHLER: {a['error']}\n" or a["text"])

  
  urls = {}
  for a in antworten:
    for u in re.findall(r"https?://[^\s\)\]>\"'|]+", a["text"]):
      u = u.rstrip(".,;:")
      urls.setdefault(u, set()).add(a["model"])
  seiten = {}
  with cf.ThreadPoolExecutor(8) as ex:
    for u, t in zip(urls, ex.map(seite_holen, urls)):
      seiten[u] = t

  pruef_input = "\n\n".join(
    f"### {u}\nGenannt von: {', '.join(sorted(urls[u]))}\nSeitentext (gekuerzt): {seiten[u][:3500]}"
    for u in list(urls)[:45]
  )
  try:
    urteil = call("/chat/completions", {
    "model": JUDGE,
    "reasoning": {"effort": "high"},
    "max_tokens": 6000,
    "usage": {"include": True},
    "messages": [{"role": "user", "content": (JUDGE_PROMPT + pruef_input)}],
    })
    bericht = urteil["choices"][0]["message"]["content"]
    judge_cost = (urteil.get("usage") or {}).get("cost", 0) or 0
  except Exception as e:
    bericht = f"Pruef-KI fehlgeschlagen ({fehlertext(e)}). Rohantworten und Seitentexte liegen trotzdem vor."
    judge_cost = 0
  kosten = sum(a["cost"] for a in antworten) + judge_cost
  with open(f"out/seitentexte_{stamp}.md", "w", encoding="utf-8") as f:
    f.write(pruef_input)
  with open(f"out/spaeher_{stamp}.md", "w", encoding="utf-8") as f:
    f.write(f"# KI-Spaeher {stamp} UTC\n\nKosten dieses Laufs: {kosten:.3f} USD, "
        f"Restguthaben vorher: {rest:.2f} USD\n\n## Gepruefte Kandidaten\n\n{bericht}\n\n")
    for a in antworten:
      f.write(f"\n---\n## Rohantwort {a['model']} ({a['cost']:.3f} USD)\n\n")
      f.write(a.get("error") and f"FEHLER: {a['error']}\n" or a["text"])
  print(f"Fertig. Kosten {kosten:.3f} USD. {len(urls)} Links geprueft.")


if __name__ == "__main__":
  main()
