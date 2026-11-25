# KI-Spaeher v4: 4 Suchwinkel x mehrere KIs ueber OpenRouter, Links selbst pruefen, Pruef-KI ordnet ein.
# Kostenbremse MIN_REST_USD. Rohantworten werden sofort gespeichert.
import concurrent.futures as cf
import datetime as dt
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

API = "https://openrouter.ai/api/v1"
KEY = os.environ.get("OPENROUTER_API_KEY", "")
MIN_REST_USD = float(os.environ.get("MIN_REST_USD", "4"))
D = os.environ.get("SPAEHER_DIR", "spaeher")
BASIS = open(f"{D}/prompt.txt", encoding="utf-8").read()
JUDGE_PROMPT = open(f"{D}/judge.txt", encoding="utf-8").read()
WINKEL = {}
for teil in open(f"{D}/winkel.txt", encoding="utf-8").read().split("### ")[1:]:
  k, _, txt = teil.partition("\n")
  WINKEL[k.strip()] = txt.strip()

# (Modell, Websuche-Plugin?, Winkel)
PANEL = [("perplexity/sonar-deep-research", False, "AD")] + [
  (m, True, "ABCD") for m in (
    "deepseek/deepseek-v4.1-flash", "qwen/qwen4-max", "z-ai/glm-5.3", "moonshotai/kimi-k3.1-pro")
]
JOBS = [(m, web, w) for m, web, ws in PANEL for w in ws if w in WINKEL]
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


def fehlertext(e):
  if isinstance(e, urllib.error.HTTPError):
    try:
      return f"HTTP {e.code}: {e.read().decode('utf-8', 'ignore')[:400]}"
    except Exception:
      return f"HTTP {e.code}"
  return str(e)[:400]


def frage(job):
  model, web, w = job
  prompt = BASIS + "\n" + WINKEL[w]
  kosten, notiz, denk = 0.0, "", ""
  # Versuch 1: hohe Denkstufe, viel Platz. Versuch 2 (nur wenn Text leer): niedrige Denkstufe.
  for effort, maxt in (("high", 32000), ("low", 16000)):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "reasoning": {"effort": effort}, "max_tokens": maxt, "usage": {"include": True}}
    if web:
      body["plugins"] = [{"id": "web", "max_results": 15}]
    r = None
    for versuch in range(2):  # bei Ueberlast (429/5xx) einmal kurz warten und wiederholen
      try:
        r = call("/chat/completions", body)
        break
      except Exception as e:
        notiz += f" [Fehler bei {effort}: {fehlertext(e)}]"
        if isinstance(e, urllib.error.HTTPError) and (e.code == 429 or e.code >= 500) and versuch == 0:
          time.sleep(25)
          continue
        break
    if r is None:
      continue
    kosten += float((r.get("usage") or {}).get("cost", 0) or 0)
    msg = (r.get("choices") or [{}])[0].get("message") or {}
    text = (msg.get("content") or "").strip()
    grund = (r.get("choices") or [{}])[0].get("finish_reason")
    if text:
      if grund == "length":
        notiz += " [Achtung: bei Laengengrenze abgeschnitten]"
      return {"model": model, "winkel": w, "text": text, "cost": kosten, "notiz": notiz}
    notiz += f" [leere Antwort bei {effort}, finish={grund}]"
    denk = (msg.get("reasoning") or "").strip()
  return {"model": model, "winkel": w, "cost": kosten, "notiz": notiz,
          "text": ("(Nur Denkprotokoll, keine Antwort:)\n" + denk[-12000:]) if denk else ""}


def seite_holen(url):
  try:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    with urllib.request.urlopen(req, timeout=25) as r:
      html = r.read(800_000).decode("utf-8", "ignore")
    text = re.sub(r"(?s)<(script|style|noscript).*?</\1>", " ", html)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text)[:15000]
  except Exception as e:
    return f"FEHLER BEIM ABRUF: {e}"


def saubere_url(u):
  u = re.sub(r"\[\d+\]?$", "", u)  # Perplexity-Zitatreste wie [3
  return u.rstrip(".,;:!?*_`]")


def schreibe_roh(pfad, antworten):
  with open(pfad, "w", encoding="utf-8") as f:
    for a in antworten:
      f.write(f"\n---\n## Rohantwort {a['model']} | Winkel {a['winkel']} ({a['cost']:.3f} USD){a['notiz']}\n\n")
      f.write(a["text"] or "(leer)")
      f.write("\n")


def main():
  rest = rest_guthaben()
  print(f"Restguthaben: {rest:.2f} USD, {len(JOBS)} Anfragen", flush=True)
  if rest < MIN_REST_USD:
    sys.exit(f"Abbruch: Restguthaben {rest:.2f} < {MIN_REST_USD} USD")

  stamp = dt.datetime.utcnow().strftime("%Y-%m-%d_%H%M")
  os.makedirs("out", exist_ok=True)
  roh = f"out/rohantworten_{stamp}.md"
  antworten = []
  with cf.ThreadPoolExecutor(10) as ex:
    for a in ex.map(frage, JOBS):
      antworten.append(a)
      print(f"{a['model']} [{a['winkel']}]: {len(a['text'])} Zeichen, {a['cost']:.3f} USD{a['notiz']}", flush=True)
      schreibe_roh(roh, antworten)  # nach jeder Antwort sichern

  urls = {}
  for a in antworten:
    for u in re.findall(r"https?://[^\s\)\]>\"'|<]+", a["text"]):
      u = saubere_url(u)
      if len(u) > 12:
        urls.setdefault(u, set()).add(f"{a['model'].split('/')[-1]}[{a['winkel']}]")
  # haeufig genannte zuerst, dann Reihenfolge des Auftauchens
  liste = sorted(urls, key=lambda u: -len(urls[u]))[:90]
  seiten = {}
  with cf.ThreadPoolExecutor(10) as ex:
    for u, t in zip(liste, ex.map(seite_holen, liste)):
      seiten[u] = t
  pruef_input = "\n\n".join(
    f"### {u}\nGenannt von: {', '.join(sorted(urls[u]))}\nSeitentext (gekuerzt): {seiten[u][:2500]}" for u in liste)
  with open(f"out/seitentexte_{stamp}.md", "w", encoding="utf-8") as f:
    f.write(pruef_input)

  judge_cost, bericht = 0.0, ""
  try:
    urteil = call("/chat/completions", {
      "model": JUDGE, "reasoning": {"effort": "medium"}, "max_tokens": 20000, "usage": {"include": True},
      "messages": [{"role": "user", "content": JUDGE_PROMPT + pruef_input}]})
    bericht = urteil["choices"][0]["message"]["content"] or ""
    if urteil["choices"][0].get("finish_reason") == "length":
      bericht += "\n\n(Achtung: Pruef-KI bei Laengengrenze abgeschnitten.)"
    judge_cost = float((urteil.get("usage") or {}).get("cost", 0) or 0)
  except Exception as e:
    bericht = f"Pruef-KI fehlgeschlagen ({fehlertext(e)}). Rohantworten und Seitentexte liegen trotzdem vor."
  kosten = sum(a["cost"] for a in antworten) + judge_cost
  with open(f"out/spaeher_{stamp}.md", "w", encoding="utf-8") as f:
    f.write(f"# KI-Spaeher v4 {stamp} UTC\n\nKosten: {kosten:.3f} USD (Pruef-KI {judge_cost:.3f}), "
            f"Restguthaben vorher: {rest:.2f} USD, {len(urls)} Links gefunden, {len(liste)} geprueft\n\n{bericht}\n")
  print(f"Fertig. Kosten {kosten:.3f} USD. {len(urls)} Links, {len(liste)} geprueft.")


if __name__ == "__main__":
  main()
