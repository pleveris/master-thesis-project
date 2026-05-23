# -*- coding: utf-8 -*-
"""
Liveness check for the QWS dataset WSDL endpoints (2026).

For each service it performs a GET on the WSDL Address and classifies the result:
  - alive=True  iff HTTP 200
  - serves_wsdl=True iff the 200 response actually looks like XML/WSDL
    (a 200 from a repurposed/parked domain returns HTML, so this separates
     genuinely-working services from URLs that merely respond)
  - reason: 200-WSDL / 200-other / HTTP-<code> / DNS-fail / timeout /
            conn-refused / SSL-error / other

Writes a per-service CSV and prints a summary. NOTE: the QWS QoS values are a
2008 snapshot; this checks current (2026) reachability only, not the validity
of the recorded measurements.

Run:  python src/real-data/check_liveness.py
"""

import os
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

warnings.filterwarnings("ignore")
requests.packages.urllib3.disable_warnings()

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(HERE, "datasets", "qws.csv")
OUT = os.path.join(HERE, "output", "liveness_check.csv")
TIMEOUT = 6
WORKERS = 40
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; QoS-liveness-check/1.0)"}


def classify(row):
    url = str(row["WSDL Address"]).strip()
    name = row["Service Name"]
    try:
        r = requests.get(url, timeout=TIMEOUT, headers=HEADERS,
                         allow_redirects=True, verify=False, stream=True)
        status = r.status_code
        serves_wsdl = False
        if status == 200:
            ct = r.headers.get("Content-Type", "").lower()
            snippet = ""
            try:
                for chunk in r.iter_content(4096):
                    snippet = chunk.decode("utf-8", "ignore").lower()
                    break
            except Exception:
                pass
            serves_wsdl = ("xml" in ct and "html" not in ct) or \
                          ("<?xml" in snippet and ("definitions" in snippet or "wsdl" in snippet))
            reason = "200-WSDL" if serves_wsdl else "200-other"
        else:
            reason = f"HTTP-{status}"
        r.close()
        return (name, url, status, status == 200, serves_wsdl, reason)
    except requests.exceptions.SSLError:
        return (name, url, None, False, False, "SSL-error")
    except (requests.exceptions.ConnectTimeout, requests.exceptions.ReadTimeout):
        return (name, url, None, False, False, "timeout")
    except requests.exceptions.ConnectionError as e:
        msg = str(e)
        if any(s in msg for s in ("Name or service not known", "getaddrinfo", "NameResolutionError")):
            return (name, url, None, False, False, "DNS-fail")
        return (name, url, None, False, False, "conn-refused")
    except Exception as e:
        return (name, url, None, False, False, type(e).__name__)


def main():
    df = pd.read_csv(DATASET)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df.columns = [c.strip() for c in df.columns]
    rows = [r for _, r in df.iterrows()]

    print(f"Checking {len(rows)} WSDL endpoints (timeout={TIMEOUT}s, workers={WORKERS})...")
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        results = list(ex.map(classify, rows))

    out = pd.DataFrame(results, columns=["Service Name", "WSDL Address", "http_status",
                                         "alive", "serves_wsdl", "reason"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT, index=False)

    n = len(out)
    alive = int(out["alive"].sum())
    wsdl = int(out["serves_wsdl"].sum())
    print(f"\n=== Liveness summary ({n} services) ===")
    print(f"  alive (HTTP 200):           {alive:>5}  ({alive/n*100:.1f}%)")
    print(f"    of which serve a WSDL:    {wsdl:>5}  ({wsdl/n*100:.1f}%)")
    print(f"  non-functioning:            {n-alive:>5}  ({(n-alive)/n*100:.1f}%)")
    print("\n  reason breakdown:")
    for reason, c in sorted(Counter(out["reason"]).items(), key=lambda kv: -kv[1]):
        print(f"    {reason:<14} {c}")
    print(f"\nPer-service status written to: {OUT}")


if __name__ == "__main__":
    main()
